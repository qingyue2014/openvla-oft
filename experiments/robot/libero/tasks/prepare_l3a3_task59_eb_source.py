"""Bind reviewed task59 EB bytes and adapt them for the evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py


PROMPT = "pick up the tomato sauce and put it in the tray"
KEY = PROMPT.replace(" ", "_")
EXPECTED_HDF5_SHA256 = "da6efc49c24513644b2138dd6ababc5abd8a57afe507a0991e5cc9955e40cdad"
EXPECTED_STATE_SHA256 = "e8156dd33774f028f8a098b9463c661e12853e589ade5a8cad77990f543eb66b"
EXPECTED_POLICY_STATUS = "PASS_L3A3_TASK59_POLICY_VIEW_REVIEWED"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--review_json", required=True)
    parser.add_argument("--out_hdf5", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    report = json.loads((candidate_dir / "report.json").read_text())
    review = json.loads(Path(args.review_json).read_text())
    source = candidate_dir / "l3a3_task59_eb_one.hdf5"
    source_digest = sha256(source.read_bytes())
    if report["verdict"] != "PASS_L3A3_TASK59_ONE_STATE_STATIC_GATE":
        raise RuntimeError("task59 static gate is not PASS")
    if report["prompt"] != PROMPT:
        raise RuntimeError("task59 prompt mismatch")
    if review["status"] != EXPECTED_POLICY_STATUS:
        raise RuntimeError("task59 independent policy-view gate is not PASS")
    if int(review["reviewed_job_id"]) != 490085:
        raise RuntimeError("task59 review is not bound to job 490085")
    if source_digest != EXPECTED_HDF5_SHA256:
        raise RuntimeError("task59 reviewed EB HDF5 byte hash mismatch")
    for name, digest in review["evidence_sha256"].items():
        path = candidate_dir / name
        if sha256(path.read_bytes()) != digest:
            raise RuntimeError(f"task59 reviewed evidence hash mismatch: {name}")

    with h5py.File(source, "r") as handle:
        state = handle[KEY]["demo_0"][:]
    state_digest = sha256(state.tobytes())
    if state_digest != EXPECTED_STATE_SHA256:
        raise RuntimeError("task59 reviewed EB state hash mismatch")
    if state_digest != review["state_sha256"]["eb"]:
        raise RuntimeError("task59 policy review references another EB state")

    output = Path(args.out_hdf5)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        task_group = handle.create_group(KEY)
        demo = task_group.create_group("demo_0")
        demo.create_dataset("initial_state", data=state)
        demo.attrs["success"] = True
        task_group.attrs["prompt"] = PROMPT
        task_group.attrs["source_hdf5_sha256"] = source_digest
        task_group.attrs["source_state_sha256"] = state_digest
        task_group.attrs["policy_review_status"] = EXPECTED_POLICY_STATUS
        task_group.attrs["policy_review_job_id"] = 490085

    binding = {
        "verdict": "PASS_L3A3_TASK59_EB_SOURCE_INPUT_BINDING",
        "task_id": 59,
        "prompt": PROMPT,
        "task_description_override": None,
        "source_hdf5_sha256": source_digest,
        "source_state_sha256": state_digest,
        "eval_hdf5": output.name,
        "eval_hdf5_sha256": sha256(output.read_bytes()),
        "policy_review_status": EXPECTED_POLICY_STATUS,
        "policy_review_job_id": 490085,
    }
    Path(args.out_report).write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )
    print(binding["verdict"])


if __name__ == "__main__":
    main()
