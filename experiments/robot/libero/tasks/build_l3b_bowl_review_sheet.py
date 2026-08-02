"""Build a compact human-review sheet from exact first policy frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONDITION_LABEL,
    INITIAL_GATE_VERDICT,
    SCENE_ID,
    TASK_PROMPT,
    sha256_path,
)


def build(manifest_path: str | Path, output_path: str | Path, rows: int) -> dict:
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verdict") != INITIAL_GATE_VERDICT or manifest.get("scenario") != SCENE_ID:
        raise ValueError("review sheet requires a valid initial gate manifest")
    if rows < 1 or rows > min(10, int(manifest["count"])):
        raise ValueError("rows must be between 1 and min(10, state count)")
    conditions = ("native", "premature_close", "prerequisite_done")
    labels = {
        "native": "EB  native: open + bowl outside",
        "premature_close": "ER  premature close: closed + bowl outside",
        "prerequisite_done": "EC  prerequisite done: open + bowl inside",
    }
    font = ImageFont.load_default()
    tile_w = tile_h = 224
    margin = 14
    header_h = 70
    label_h = 30
    canvas = Image.new(
        "RGB",
        (margin * 2 + len(conditions) * tile_w, header_h + rows * (label_h + tile_h) + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), "L3-B bowl order — exact pi0.5 agent-view first-policy frames", fill="black", font=font)
    draw.text((margin, 30), TASK_PROMPT, fill="black", font=font)
    draw.text((margin, 48), "Human gate: verify drawer state, bowl location, plausibility, and visibility.", fill="black", font=font)
    bound_images = []
    for row in range(rows):
        episode = manifest["episodes"][row]
        y = header_h + row * (label_h + tile_h)
        for column, condition in enumerate(conditions):
            x = margin + column * tile_w
            draw.rectangle((x, y, x + tile_w - 1, y + label_h - 1), fill=(235, 235, 235))
            draw.text((x + 4, y + 8), f"state {row:02d}  {labels[condition]}", fill="black", font=font)
            path = Path(
                episode["conditions"][condition]["policy_images"]["agentview_pi05_224"]
            ).resolve(strict=True)
            image = Image.open(path).convert("RGB")
            if image.size != (tile_w, tile_h):
                raise ValueError(f"unexpected policy image size: {path} {image.size}")
            canvas.paste(image, (x, y + label_h))
            bound_images.append({"path": str(path), "sha256": sha256_path(path)})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    record = {
        "scenario": SCENE_ID,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_path(manifest_path),
        "rows": rows,
        "conditions": [CONDITION_LABEL[value] for value in conditions],
        "images": bound_images,
        "review_sheet": str(output.resolve()),
        "review_sheet_sha256": sha256_path(output),
        "human_review_approved": False,
        "formal_authorized": False,
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    result = build(args.manifest, args.output, args.rows)
    print(f"PASS_L3B_BOWL_REVIEW_SHEET {result['review_sheet']}")


if __name__ == "__main__":
    main()
