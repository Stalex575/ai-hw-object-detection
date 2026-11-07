"""Collect the small result files of each run from outputs/ (git-ignored) into experiments/,
draw the Faster R-CNN validation examples, and plot the summary figures.

Run from the repo root:  .venv/bin/python experiments/make_summary.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.viz_gt import grid  # noqa: E402

OUT, EXP = REPO / "outputs", REPO / "experiments"

# Fixed colour per run (categorical slots 1-4 of the default dataviz palette), same in every figure.
RUNS = {
    "yolo26s_smoke_fold0_640_s42_v1": ("YOLO26s 640 (smoke)", "#2a78d6"),
    "yolo26s_base_fold0_1280_e25_s42_v1": ("YOLO26s 1280", "#eb6834"),
    "rtdetrl_base_fold0_1280_e12_s42_v1": ("RT-DETR-L 1280", "#1baf7a"),
    "frcnn_base_fold0_1280_e12_b2_s42_v1": ("Faster R-CNN 1280", "#eda100"),
}
KEEP = ["metrics.json", "experiment.csv", "speed.json", "runtime.json", "training.json", "preflight.json",
        "config.json", "train_log.csv", "validation_examples.jpg", "submission_check.json"]
KEEP_TRAIN = ["args.yaml", "results.csv", "results.png"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def copy_results():
    for run in RUNS:
        dst = EXP / run
        dst.mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (OUT / run / name).exists():
                shutil.copy2(OUT / run / name, dst / name)
        for name in KEEP_TRAIN:
            if (OUT / run / "train" / name).exists():
                shutil.copy2(OUT / run / "train" / name, dst / name)


def frcnn_examples(run="frcnn_base_fold0_1280_e12_b2_s42_v1"):
    """Same frames and drawing rules as the YOLO notebook: first frame of the first 4 val clips,
    GT green, predictions with score >= 0.1 red."""
    gt = json.loads((REPO / "prep/coco/train_step5_fold0_val.json").read_text())
    predictions = json.loads((OUT / run / "val_predictions.json").read_text())
    tiles, clips = [], set()
    for image in gt["images"]:
        clip = Path(image["file_name"]).parent.name
        if clip in clips:
            continue
        clips.add(clip)
        picture = Image.open(REPO / "train" / image["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(picture)
        targets = [a for a in gt["annotations"] if a["image_id"] == image["id"] and not a.get("iscrowd")]
        detected = [p for p in predictions if p["image_id"] == image["id"] and p["score"] >= 0.1]
        for records, color in ((targets, "lime"), (detected, "red")):
            for record in records:
                x, y, w, h = record["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text((10, 10), clip + " | green: GT, red: predictions", fill="white", stroke_width=2, stroke_fill="black")
        tiles.append(picture)
        if len(tiles) == 4:
            break
    grid(tiles, cols=2, tile_w=640).save(EXP / run / "validation_examples.jpg", quality=90)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves():
    """Per-epoch validation mAP@[.5:.95] on the fold-0 val frames. Two panels, because the
    evaluators differ: Ultralytics' built-in validator vs our shared evaluator."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), sharey=True, facecolor=SURFACE)
    for run in list(RUNS)[:3]:
        label, color = RUNS[run]
        rows = list(csv.DictReader((EXP / run / "results.csv").open()))
        x = [int(r["epoch"]) for r in rows]
        y = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        left.plot(x, y, color=color, linewidth=2, marker="o", markersize=3.5, label=label)
        left.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=8.5, color=INK)
    run = list(RUNS)[3]
    label, color = RUNS[run]
    rows = [r for r in csv.DictReader((EXP / run / "train_log.csv").open()) if r.get("mAP")]
    x = [int(r["epoch"]) for r in rows]
    y = [float(r["mAP"]) for r in rows]
    right.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
    right.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=INK)
    left.set_title("Ultralytics runs - Ultralytics validator", fontsize=10.5, color=INK, loc="left")
    right.set_title("Faster R-CNN - shared evaluator (every 2 epochs)", fontsize=10.5, color=INK, loc="left")
    left.set_ylabel("val mAP@[.5:.95], fold 0", fontsize=9.5, color=MUTED)
    for ax in (left, right):
        style(ax)
        ax.set_xlabel("epoch", fontsize=9.5, color=MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    right.set_xlim(0, 13)
    fig.tight_layout()
    fig.savefig(EXP / "figures" / "val_map_per_epoch.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def per_class():
    """Shared-evaluator mAP per class for the three 1280 baselines (grouped horizontal bars)."""
    runs = list(RUNS)[1:]
    classes = list(json.loads((EXP / runs[0] / "metrics.json").read_text())["per_class"])
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    height = 0.26
    for k, run in enumerate(runs):
        label, color = RUNS[run]
        values = json.loads((EXP / run / "metrics.json").read_text())["per_class"]
        ys = [i + (k - 1) * (height + 0.02) for i in range(len(classes))]
        ax.barh(ys, [values[c]["mAP"] for c in classes], height=height, color=color, label=label,
                edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("mAP@[.5:.95], fold 0, shared evaluator", fontsize=9.5, color=MUTED)
    ax.set_title("Per-class mAP of the 1280 baselines", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP / "figures" / "per_class_map.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_csv():
    rows = []
    for run, (label, _) in RUNS.items():
        rows.extend(csv.DictReader((EXP / run / "experiment.csv").open()))
    with (EXP / "experiments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    (EXP / "figures").mkdir(parents=True, exist_ok=True)
    copy_results()
    frcnn_examples()
    curves()
    per_class()
    summary_csv()
    print("done")
