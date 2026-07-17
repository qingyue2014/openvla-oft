#!/usr/bin/env python3
"""Auditable SSH/Slurm runner for remote PhysCog scene validation.

This module intentionally exposes a small phase registry instead of arbitrary
remote shell execution.  Scene-editing agents can therefore run expensive
MuJoCo checks on Superpod while retaining a local ledger of exactly what ran.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PhaseSpec:
    command: tuple[str, ...]
    count_env: str | None = None
    artifacts: tuple[str, ...] = ()


PHASES: Mapping[tuple[str, str], PhaseSpec] = {
    ("l1a2", "check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1a_evals.sh", "l1a2_check"),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1a2_task1_upright_cookie_pairing.json",
        ),
    ),
    ("l1a2", "preview"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1a_evals.sh", "l1a2_preview"),
        artifacts=("experiments/robot/libero/tasks/l1a2_preview",),
    ),
    ("l1a2", "safe_reference"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1a_evals.sh",
            "l1a2_safe_reference",
        ),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l1a2_safe_reference.md",
            "experiments/logs/l1a2_safe_reference.csv",
        ),
    ),
    ("l1a2", "smoke"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1a_evals.sh", "l1a2_smoke"),
        count_env="SMOKE_TRIALS",
        artifacts=("experiments/logs/l1a_results.md",),
    ),
}


VERDICT_RE = re.compile(
    r"(?:Verdict:\s*(?:\*\*)?|verdict=|\"occlusion_gate\"\s*:\s*\")"
    r"([A-Z][A-Z0-9_-]+)",
    re.IGNORECASE,
)
MARKER_RE = re.compile(r"^__PHYSCOG_([A-Z_]+)__=(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    user: str
    control_socket: str
    remote_repo: str
    remote_python_bin: str
    branch: str
    account: str
    partition: str
    nodes: int
    gpus: int
    time_limit: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def build_compute_command(
    cfg: RemoteConfig, spec: PhaseSpec, count: int
) -> str:
    env = []
    if spec.count_env:
        env.append(f"export {spec.count_env}={shlex.quote(str(count))}")
    steps = [
        f"cd {shlex.quote(cfg.remote_repo)}",
        f"export PATH={shlex.quote(cfg.remote_python_bin)}:$PATH",
        *env,
        "printf '__PHYSCOG_COMPUTE_NODE__=%s\\n' \"$(hostname)\"",
        shell_join(spec.command),
    ]
    payload = " && ".join(steps)
    return shell_join(
        (
            "srun",
            "--account",
            cfg.account,
            "--partition",
            cfg.partition,
            "--nodes",
            str(cfg.nodes),
            "--gpus",
            str(cfg.gpus),
            "--time",
            cfg.time_limit,
            "bash",
            "-lc",
            payload,
        )
    )


def build_remote_script(
    cfg: RemoteConfig, spec: PhaseSpec, count: int, sync: bool = True
) -> str:
    lines = ["set -euo pipefail", f"cd {shlex.quote(cfg.remote_repo)}"]
    if sync:
        lines.extend(
            (
                shell_join(("git", "fetch", "origin", cfg.branch)),
                shell_join(("git", "checkout", cfg.branch)),
                shell_join(("git", "pull", "--ff-only", "origin", cfg.branch)),
            )
        )
    lines.extend(
        (
            "printf '__PHYSCOG_LOGIN_NODE__=%s\\n' \"$(hostname)\"",
            "printf '__PHYSCOG_COMMIT__=%s\\n' \"$(git rev-parse HEAD)\"",
            build_compute_command(cfg, spec, count),
        )
    )
    return "\n".join(lines)


def ssh_argv(cfg: RemoteConfig, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-S",
        cfg.control_socket,
        "-o",
        "BatchMode=yes",
        cfg.target,
        remote_script,
    ]


def extract_verdicts(text: str) -> list[str]:
    verdicts: list[str] = []
    for match in VERDICT_RE.finditer(text):
        token = match.group(1).upper()
        if token not in verdicts:
            verdicts.append(token)
    return verdicts


def classify_result(returncode: int, text: str, verdicts: Sequence[str]) -> str:
    validator_signatures = (
        "Traceback (most recent call last)",
        "KeyError:",
        "NameError:",
        "AttributeError:",
        "UnboundLocalError:",
    )
    infrastructure_signatures = (
        "Permission denied",
        "Control socket connect",
        "Could not resolve hostname",
        "ModuleNotFoundError:",
        "Fatal Python error: Aborted",
        "srun: error:",
        "Unable to allocate resources",
        "Repository Not Found",
    )
    if any(signature in text for signature in validator_signatures):
        return "validator_bug"
    if any(signature in text for signature in infrastructure_signatures):
        return "infrastructure_failure"
    if any(v.startswith(("FAIL", "NEEDS_", "BENCHMARK_INCOMPLETE")) for v in verdicts):
        return "gate_failure"
    if returncode != 0:
        return "command_failure"
    if any(v.startswith(("PASS", "BENCHMARK_READY")) for v in verdicts):
        return "pass"
    return "completed"


def parse_markers(text: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(2).strip() for match in MARKER_RE.finditer(text)}


def _stream_command(argv: Sequence[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            chunks.append(line)
        return process.wait(), "".join(chunks)


def _local_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_dir(root: Path, scenario: str, phase: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"{stamp}-{scenario}-{phase}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{scenario}-{phase}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _fetch_artifact(cfg: RemoteConfig, remote_path: str, output_root: Path) -> bool:
    destination = output_root / remote_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = f"{cfg.target}:{cfg.remote_repo.rstrip('/')}/{remote_path}"
    argv = [
        "scp",
        "-q",
        "-r",
        "-o",
        f"ControlPath={cfg.control_socket}",
        source,
        str(destination),
    ]
    return subprocess.run(argv, check=False).returncode == 0


def _artifact_texts(root: Path, artifacts: Sequence[str]) -> str:
    texts: list[str] = []
    for artifact in artifacts:
        path = root / artifact
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".txt", ".csv"}:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def _config_from_args(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        host=args.host,
        user=args.user,
        control_socket=args.control_socket,
        remote_repo=args.remote_repo,
        remote_python_bin=args.remote_python_bin,
        branch=args.branch,
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        gpus=args.gpus,
        time_limit=args.time_limit,
    )


def command_probe(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    remote = " && ".join(
        (
            "printf '__PHYSCOG_LOGIN_NODE__=%s\\n' \"$(hostname)\"",
            f"cd {shlex.quote(cfg.remote_repo)}",
            "printf '__PHYSCOG_COMMIT__=%s\\n' \"$(git rev-parse HEAD)\"",
            "printf '__PHYSCOG_BRANCH__=%s\\n' \"$(git branch --show-current)\"",
        )
    )
    argv = ssh_argv(cfg, remote)
    if args.dry_run:
        print(shell_join(argv))
        return 0
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    print(output, end="")
    if args.json:
        print(json.dumps(parse_markers(output), indent=2, sort_keys=True))
    return result.returncode


def command_run(args: argparse.Namespace) -> int:
    key = (args.scenario.lower(), args.phase.lower())
    if key not in PHASES:
        choices = ", ".join(f"{s}:{p}" for s, p in sorted(PHASES))
        raise SystemExit(f"Unregistered phase {key[0]}:{key[1]}; choose one of: {choices}")
    spec = PHASES[key]
    cfg = _config_from_args(args)
    remote_script = build_remote_script(cfg, spec, args.count, sync=not args.no_sync)
    argv = ssh_argv(cfg, remote_script)
    if args.dry_run:
        print(shell_join(argv))
        return 0

    run_dir = _run_dir(Path(args.state_root), *key)
    print(f"[physcog-agent] run ledger: {run_dir}")
    returncode, output = _stream_command(argv, run_dir / "remote.log")

    fetched: list[str] = []
    missing: list[str] = []
    for artifact in spec.artifacts:
        if _fetch_artifact(cfg, artifact, run_dir / "artifacts"):
            fetched.append(artifact)
        else:
            missing.append(artifact)
    evidence = output + "\n" + _artifact_texts(run_dir / "artifacts", fetched)
    verdicts = extract_verdicts(evidence)
    classification = classify_result(returncode, evidence, verdicts)
    ledger = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario": key[0],
        "phase": key[1],
        "count": args.count,
        "classification": classification,
        "verdicts": verdicts,
        "returncode": returncode,
        "local_commit": _local_commit(),
        "remote_markers": parse_markers(output),
        "fetched_artifacts": fetched,
        "missing_artifacts": missing,
        "remote_config": {**asdict(cfg), "control_socket": cfg.control_socket},
        "registered_command": list(spec.command),
        "count_env": spec.count_env,
    }
    (run_dir / "run.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[physcog-agent] classification={classification} "
        f"verdicts={','.join(verdicts) or '--'} artifacts={len(fetched)}/{len(spec.artifacts)}"
    )
    if classification in {"pass", "completed"}:
        return 0
    if classification == "gate_failure":
        return 2
    return returncode or 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("PHYSCOG_HOST", "superpod.ust.hk"))
    parser.add_argument("--user", default=os.environ.get("PHYSCOG_USER", "drwqyhappy"))
    parser.add_argument(
        "--control-socket",
        default=os.environ.get("PHYSCOG_CONTROL_SOCKET", "/tmp/physcog-superpod.sock"),
    )
    parser.add_argument(
        "--remote-repo",
        default=os.environ.get(
            "PHYSCOG_REMOTE_REPO", "/home/drwqyhappy/04-mycode/openvla-oft"
        ),
    )
    parser.add_argument(
        "--remote-python-bin",
        default=os.environ.get(
            "PHYSCOG_REMOTE_PYTHON_BIN", "/home/drwqyhappy/.conda/envs/openvla_oft/bin"
        ),
    )
    parser.add_argument("--branch", default="physcog-libero-l1")
    parser.add_argument("--account", default="trllmout")
    parser.add_argument("--partition", default="normal")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    probe = subparsers.add_parser("probe", help="Check the existing SSH control connection")
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(func=command_probe)

    run = subparsers.add_parser("run", help="Run one registered validation phase through Slurm")
    run.add_argument("--scenario", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--count", type=int, default=8)
    run.add_argument("--no-sync", action="store_true", help="Do not fast-forward the remote checkout")
    run.add_argument("--state-root", default=".physcog-agent/runs")
    run.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "count", 1) < 1:
        raise SystemExit("--count must be positive")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
