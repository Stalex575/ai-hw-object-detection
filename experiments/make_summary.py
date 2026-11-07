"""Collect the small result files of each run from outputs/ (git-ignored) into experiments/,
draw validation examples for the torchvision baselines, and plot the summary figures.

Team protocol rule: every run is reported on its LAST epoch.
Ultralytics runs were first scored on best.pt; their last.pt re-evaluation lives in outputs/<run>/last/
and is what the tables and figures use. Runs whose outputs are not finished yet are skipped.

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

# Colour follows the family (categorical slots of the default dataviz palette, fixed order).
FAMILY = {
    "YOLO26s": "#eb6834",
    "RT-DETR-L": "#1baf7a",
    "Faster R-CNN": "#eda100",
    "RetinaNet": "#e87ba4",
}
# Within one family's curve panel, variants take slots 1..4 in this order (baseline first).
VARIANT_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# run -> (family, kind, short label of the single change; "baseline" for the family baseline)
RUNS = {
    "yolo26s_smoke_fold0_640_s42_v1": ("YOLO26s", "ultralytics", "640 smoke, 20 ep"),
    "yolo26s_base_fold0_1280_e25_s42_v1": ("YOLO26s", "ultralytics", "baseline"),
    "rtdetrl_base_fold0_1280_e12_s42_v1": ("RT-DETR-L", "ultralytics", "baseline"),
    "frcnn_base_fold0_1280_e12_b2_s42_v1": ("Faster R-CNN", "torchvision", "baseline"),
    "frcnn_anchors16-256_fold0_1280_e12_b2_s42_v1": ("Faster R-CNN", "torchvision", "RPN anchors 16-256"),
    "frcnn_giou_fold0_1280_e12_b2_s42_v1": ("Faster R-CNN", "torchvision", "RoI GIoU loss"),
    "retinanet_base_fold0_1280_e12_b2_s42_v1": ("RetinaNet", "torchvision", "baseline"),
    "retinanet_kmeans_fold0_1280_e12_b2_s42_v1": ("RetinaNet", "torchvision", "k-means anchors, per level"),
    "retinanet_gamma1.5_fold0_1280_e12_b2_s42_v1": ("RetinaNet", "torchvision", "focal gamma 1.5"),
    "frcnn_posfrac0.5_fold0_1280_e12_b2_s42_v1": ("Faster R-CNN", "torchvision", "RoI positive fraction 0.5"),
    "retinanet_iou0.4-0.3_fold0_1280_e12_b2_s42_v1": ("RetinaNet", "torchvision", "IoU thresholds 0.4/0.3"),
    "retinanet_kmeansshared_fold0_1280_e12_b2_s42_v1": ("RetinaNet", "torchvision", "k-means anchors, shared"),
}
EXAMPLES = ["frcnn_base_fold0_1280_e12_b2_s42_v1", "retinanet_base_fold0_1280_e12_b2_s42_v1"]
KEEP = ["metrics.json", "experiment.csv", "speed.json", "runtime.json", "training.json", "preflight.json",
        "config.json", "train_log.csv", "validation_examples.jpg", "submission_check.json"]
KEEP_TRAIN = ["args.yaml", "results.csv", "results.png"]
KEEP_LAST = ["metrics.json", "experiment.csv", "speed.json", "submission_check.json"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def finished(run):
    return (OUT / run / "experiment.csv").exists()


def label(run):
    family, _, change = RUNS[run]
    return family if change == "baseline" else f"{family}: {change}"


def copy_results():
    for run in RUNS:
        if not finished(run):
            continue
        dst = EXP / run
        dst.mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (OUT / run / name).exists():
                shutil.copy2(OUT / run / name, dst / name)
        for name in KEEP_TRAIN:
            if (OUT / run / "train" / name).exists():
                shutil.copy2(OUT / run / "train" / name, dst / name)
        for name in KEEP_LAST:
            if (OUT / run / "last" / name).exists():
                (dst / "last").mkdir(exist_ok=True)
                shutil.copy2(OUT / run / "last" / name, dst / "last" / name)


def reported(run, name):
    """The protocol file of a run: <run>/last/<name> for Ultralytics runs once re-scored, else <run>/<name>."""
    last = EXP / run / "last" / name
    return last if last.exists() else EXP / run / name


def examples(run):
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
    """Per-epoch validation mAP@[.5:.95] on the fold-0 val frames. Ultralytics runs: Ultralytics' validator
    (not comparable with the shared evaluator). Torchvision runs: shared evaluator, one panel per family."""
    ultra = [r for r in RUNS if RUNS[r][1] == "ultralytics" and (EXP / r / "results.csv").exists()]
    tv_families = [f for f in FAMILY if any(RUNS[r][0] == f and RUNS[r][1] == "torchvision" and finished(r)
                                            for r in RUNS)]
    panels = 1 + len(tv_families)
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4), sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]

    ends = {}  # axis -> [(x, y)] of line ends, labelled after all lines are drawn

    def draw(ax, x, y, color, text):
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=text)
        ends.setdefault(ax, []).append((x[-1], y[-1]))

    def label_ends(ax, gap=0.011):
        """End-of-line values, nudged apart vertically so close finishes (e.g. 0.272 / 0.275) stay readable."""
        placed = []
        for x, y in sorted(ends.get(ax, []), key=lambda e: e[1]):
            placed.append(max(y, placed[-1] + gap) if placed else y)
            ax.annotate(f"{y:.3f}", (x, y), xytext=(x + 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3, placed[-1]),
                        textcoords="data", va="center", fontsize=8.5, color=INK)

    for run in ultra:
        rows = list(csv.DictReader((EXP / run / "results.csv").open()))
        family, _, change = RUNS[run]
        color = FAMILY[family] if change == "baseline" else "#2a78d6"
        draw(axes[0], [int(r["epoch"]) for r in rows], [float(r["metrics/mAP50-95(B)"]) for r in rows],
             color, label(run))
    axes[0].set_title("Ultralytics runs - Ultralytics validator", fontsize=10.5, color=INK, loc="left")
    for ax, family in zip(axes[1:], tv_families):
        runs = [r for r in RUNS if RUNS[r][0] == family and finished(r)]
        for slot, run in enumerate(runs):
            rows = [r for r in csv.DictReader((EXP / run / "train_log.csv").open()) if r.get("mAP")]
            draw(ax, [int(r["epoch"]) for r in rows], [float(r["mAP"]) for r in rows], VARIANT_SLOTS[slot],
                 RUNS[run][2])
        ax.set_title(f"{family} - shared evaluator", fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(0, 13.5)
    axes[0].set_ylabel("val mAP@[.5:.95], fold 0", fontsize=9.5, color=MUTED)
    for ax in axes:
        style(ax)
        ax.set_xlabel("epoch", fontsize=9.5, color=MUTED)
        label_ends(ax)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP / "figures" / "val_map_per_epoch.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ablation():
    """Final fold-0 mAP of every run (last epoch), grouped by family, coloured by family."""
    runs = [r for r in RUNS if finished(r)]
    values = [json.loads(reported(r, "metrics.json").read_text())["metrics"]["mAP"] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(runs) + 1.2), facecolor=SURFACE)
    ax.barh(range(len(runs)), values, height=0.62, color=[FAMILY[RUNS[r][0]] for r in runs],
            edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(values):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(runs)), [label(r) for r in runs])
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("mAP@[.5:.95], fold 0, shared evaluator, last epoch", fontsize=9.5, color=MUTED)
    ax.set_title("All fold-0 runs", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(EXP / "figures" / "fold0_runs_map.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def per_class():
    """Shared-evaluator mAP per class for the 1280 family baselines (grouped horizontal bars)."""
    runs = [r for r in RUNS if RUNS[r][2] == "baseline" and finished(r)]
    classes = list(json.loads(reported(runs[0], "metrics.json").read_text())["per_class"])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    height = 0.8 / len(runs)
    for k, run in enumerate(runs):
        values = json.loads(reported(run, "metrics.json").read_text())["per_class"]
        ys = [i + (k - (len(runs) - 1) / 2) * height for i in range(len(classes))]
        ax.barh(ys, [values[c]["mAP"] for c in classes], height=height, color=FAMILY[RUNS[run][0]],
                label=RUNS[run][0], edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("mAP@[.5:.95], fold 0, shared evaluator, last epoch", fontsize=9.5, color=MUTED)
    ax.set_title("Per-class mAP of the 1280 baselines", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP / "figures" / "per_class_map.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_csv():
    """One row per run, last-epoch numbers. Ultralytics best.pt rows are kept with an @best suffix for reference.
    Keeps public_LB values already typed into experiments.csv."""
    old = {}
    if (EXP / "experiments.csv").exists():
        old = {r["id"]: r for r in csv.DictReader((EXP / "experiments.csv").open())}
    rows = []
    for run in RUNS:
        if not finished(run):
            continue
        main = next(csv.DictReader(reported(run, "experiment.csv").open()))
        main["id"] = run
        main["checkpoint"] = "last"
        rows.append(main)
        if (EXP / run / "last" / "experiment.csv").exists():
            best = next(csv.DictReader((EXP / run / "experiment.csv").open()))
            best["id"] = run + "@best"
            best["checkpoint"] = "best (Ultralytics pick on the val frames; optimistic)"
            rows.append(best)
    for row in rows:
        if old.get(row["id"], {}).get("public_LB"):
            row["public_LB"] = old[row["id"]]["public_LB"]
    with (EXP / "experiments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    (EXP / "figures").mkdir(parents=True, exist_ok=True)
    copy_results()
    for run in EXAMPLES:
        if finished(run) and not (EXP / run / "validation_examples.jpg").exists():
            examples(run)
    curves()
    ablation()
    per_class()
    summary_csv()
    print("done")
