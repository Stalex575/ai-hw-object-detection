"""k-means anchors (1 - IoU distance) for RetinaNet P3-P7, fitted on fold-0 TRAIN boxes only.

Boxes are rescaled to the nominal training scale (short side 720, long side <= 1280), i.e. the pixel space
the anchors live in. k = 15 shapes are clustered, sorted by area and split 3 per level (P3 smallest), and each
shape gets torchvision's 3 octave scales (x1, x2^(1/3), x2^(2/3)), so every level keeps 9 anchors per location
and the pretrained RetinaNet heads still fit. Also written: per-level k-means (L4; broke the shared head, see
experiments/README.md) and scale-equivariant "shared" k-means (L4b: one set of 9 shapes x each level's base).

The diagnostic is grid-aware: anchors are placed where torchvision places them (centres on the stride grid of
each level), and for every GT box we record the best IoU over all anchors of its image. A GT whose best IoU is
< 0.5 only gets a positive anchor through the low-quality-match fallback.

    python -m src.anchors_kmeans --out experiments/anchors_kmeans
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from .evaluate import read_json, write_json

REPO = Path(__file__).resolve().parents[1]
STRIDES = (8, 16, 32, 64, 128)
OCTAVES = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))


def load_boxes(coco_json, short=720, long=1280):
    coco = read_json(coco_json)
    images = {i["id"]: i for i in coco["images"]}
    per_image = {i: [] for i in images}
    for a in coco["annotations"]:
        if a.get("iscrowd") or a.get("ignore") or a["bbox"][2] < 1 or a["bbox"][3] < 1:
            continue
        info = images[a["image_id"]]
        s = min(short / min(info["height"], info["width"]), long / max(info["height"], info["width"]))
        x, y, w, h = a["bbox"]
        per_image[a["image_id"]].append([x * s, y * s, (x + w) * s, (y + h) * s])
    sizes = {}
    for i, info in images.items():
        s = min(short / min(info["height"], info["width"]), long / max(info["height"], info["width"]))
        sizes[i] = (round(info["height"] * s), round(info["width"] * s))
    return per_image, sizes


def wh_iou(wh, centers):
    inter = np.minimum(wh[:, None, 0], centers[None, :, 0]) * np.minimum(wh[:, None, 1], centers[None, :, 1])
    return inter / (wh[:, None].prod(-1) + centers[None].prod(-1) - inter)


def kmeans_iou(wh, k, seed=42, iters=300):
    """Lloyd's k-means with d = 1 - IoU of origin-aligned boxes; k-means++ style seeding; median update."""
    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(len(wh), 1)]
    while len(centers) < k:
        d = 1 - wh_iou(wh, centers).max(1)
        centers = np.vstack([centers, wh[rng.choice(len(wh), 1, p=d ** 2 / (d ** 2).sum())]])
    for _ in range(iters):
        assign = wh_iou(wh, centers).argmax(1)
        new = np.array([np.median(wh[assign == j], 0) if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    fitness = wh_iou(wh, centers).max(1).mean()
    return centers[np.argsort(centers.prod(1))], float(fitness)


def levels_from_sizes(base_sizes, ratios=(0.5, 1.0, 2.0)):
    """torchvision layout: per level, ratio (h/w) x octave scale."""
    out = []
    for base in base_sizes:
        level = []
        for r in ratios:
            for o in OCTAVES:
                size = int(base * o)
                level.append([size / math.sqrt(r), size * math.sqrt(r)])
        out.append(level)
    return out


def levels_from_shapes(shapes):
    """15 shapes sorted by area -> 3 per level, each with 3 octave scales -> 9 anchors per level."""
    return [[[float(w * o), float(h * o)] for w, h in shapes[3 * lv:3 * lv + 3] for o in OCTAVES]
            for lv in range(len(STRIDES))]


def levels_from_level_kmeans(wh, bases=(16, 32, 64, 128, 256), min_boxes=200, seed=42):
    """Stride-aware variant: assign each box to the level whose octave range it falls in (boundaries at the
    geometric midpoint between a level's largest octave base*2^(2/3) and the next level's base), then fit
    9 k-means shapes per level. Levels with < min_boxes boxes keep the torchvision layout for their base."""
    sqrt_area = np.sqrt(wh.prod(1))
    edges = [b * 2 ** (5 / 6) for b in bases[:-1]]
    level = np.searchsorted(edges, sqrt_area)
    out, info = [], []
    for lv, base in enumerate(bases):
        members = wh[level == lv]
        if len(members) >= min_boxes:
            shapes, fit = kmeans_iou(members, 9, seed)
            out.append([[float(w), float(h)] for w, h in shapes])
            info.append({"level": f"P{lv + 3}", "boxes": int(len(members)), "kmeans_fitness": fit})
        else:
            out.append(levels_from_sizes([base])[0])
            info.append({"level": f"P{lv + 3}", "boxes": int(len(members)), "kmeans_fitness": None})
    return out, info


def levels_from_shared_kmeans(wh, bases=(16, 32, 64, 128, 256), seed=42):
    """Scale-equivariant variant for RetinaNet's head, which is SHARED across P3-P7: anchor slot j must mean the
    same shape relative to the stride on every level (as in torchvision). Boxes are assigned to levels as in
    levels_from_level_kmeans, divided by their level's base size, and ONE set of 9 shapes is fitted on the
    normalised boxes; each level gets those shapes times its base. The shapes are put in the slots of the
    pretrained head's default anchors (ratio x octave, normalised) by maximum-IoU assignment, so the
    pretrained per-slot weights start from the closest shape."""
    from scipy.optimize import linear_sum_assignment
    sqrt_area = np.sqrt(wh.prod(1))
    level = np.searchsorted([b * 2 ** (5 / 6) for b in bases[:-1]], sqrt_area)
    norm = wh / np.asarray(bases, dtype=np.float64)[level][:, None]
    shapes, fit = kmeans_iou(norm, 9, seed)
    default = np.array(levels_from_sizes([1000])[0]) / 1000  # torchvision slot order, base 1
    rows, cols = linear_sum_assignment(-wh_iou(default, shapes))
    shapes = shapes[cols[np.argsort(rows)]]
    info = {"normalised_shapes": shapes.round(3).tolist(), "kmeans_fitness": fit,
            "slot_iou_with_default": wh_iou(default, shapes).diagonal().round(3).tolist(),
            "boxes_per_level": np.bincount(level, minlength=len(bases)).tolist()}
    return [[[float(w * b), float(h * b)] for w, h in shapes] for b in bases], info


def grid_anchors(wh_per_level, height, width):
    """Anchor boxes exactly as torchvision places them (batch padded to a multiple of 32)."""
    ph, pw = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
    boxes = []
    for stride, level in zip(STRIDES, wh_per_level):
        cell = torch.tensor([[-w / 2, -h / 2, w / 2, h / 2] for w, h in level]).round()
        ys = torch.arange(math.ceil(ph / stride)) * stride
        xs = torch.arange(math.ceil(pw / stride)) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        shifts = torch.stack([xx, yy, xx, yy], -1).reshape(-1, 1, 4).float()
        boxes.append((shifts + cell[None]).reshape(-1, 4))
    return torch.cat(boxes)


def best_ious(wh_per_level, per_image, sizes, image_ids):
    out = []
    for i in image_ids:
        if not per_image[i]:
            continue
        gt = torch.tensor(per_image[i])
        out.append(box_iou(gt, grid_anchors(wh_per_level, *sizes[i])).max(1).values)
    return torch.cat(out).numpy()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default=str(REPO / "prep/coco/train_step5_fold0_train.json"))
    p.add_argument("--out", default=str(REPO / "experiments/anchors_kmeans"))
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--sample", type=int, default=400, help="train images for the grid-aware IoU diagnostic")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_image, sizes = load_boxes(args.gt)
    boxes = np.array([b for v in per_image.values() for b in v])
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    shapes, fitness = kmeans_iou(wh, args.k)
    variants = {
        "default_32-512": levels_from_sizes([32, 64, 128, 256, 512]),
        "sizes_16-256": levels_from_sizes([16, 32, 64, 128, 256]),
        f"kmeans{args.k}_by_area": levels_from_shapes(shapes),
    }
    variants["kmeans_per_level"], level_info = levels_from_level_kmeans(wh)
    variants["kmeans_shared"], shared_info = levels_from_shared_kmeans(wh)
    rng = np.random.default_rng(0)
    ids = sorted(rng.choice(sorted(per_image), min(args.sample, len(per_image)), replace=False).tolist())
    sqrt_area = None
    report = {"train_boxes": int(len(wh)), "kmeans_shapes_wh": shapes.round(1).tolist(),
              "kmeans_mean_best_shape_iou": fitness, "per_level_kmeans": level_info, "shared_kmeans": shared_info, "diagnostic_images": len(ids), "variants": {}}
    hist = {}
    for name, levels in variants.items():
        ious = best_ious(levels, per_image, sizes, ids)
        if sqrt_area is None:
            gt = np.array([b for i in ids for b in per_image[i]])
            sqrt_area = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        buckets = {"<16px": sqrt_area < 16, "16-32px": (sqrt_area >= 16) & (sqrt_area < 32),
                   "32-64px": (sqrt_area >= 32) & (sqrt_area < 64), ">=64px": sqrt_area >= 64}
        report["variants"][name] = {
            "anchors_wh": [[[round(w, 1), round(h, 1)] for w, h in lv] for lv in levels],
            "mean_best_iou": float(ious.mean()), "frac_best_iou_ge_0.5": float((ious >= 0.5).mean()),
            "frac_best_iou_ge_0.4": float((ious >= 0.4).mean()),
            "frac_ge_0.5_by_size": {k: float((ious[m] >= 0.5).mean()) for k, m in buckets.items()},
            "gt_by_size": {k: int(m.sum()) for k, m in buckets.items()}}
        hist[name] = ious
        print(name, {k: v for k, v in report["variants"][name].items() if k != "anchors_wh"})
    for name in (f"kmeans{args.k}_by_area", "kmeans_per_level", "kmeans_shared"):
        write_json(out / f"anchors_{name}.json", {"wh": variants[name], "source": "fold-0 train boxes, 720/1280 scale"})
    write_json(out / "report.json", report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    for n, ious in hist.items():
        ax.hist(ious, bins=bins, histtype="step", linewidth=1.8,
                label=f"{n}: mean {ious.mean():.2f}, >=0.5 {(ious >= 0.5).mean():.0%}")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("best IoU between a GT box and any anchor (grid-aware)")
    ax.set_ylabel("GT boxes")
    ax.set_title(f"Fold-0 train, {len(ids)} images, {len(hist[name])} boxes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "anchor_best_iou_hist.png", dpi=130)


if __name__ == "__main__":
    main()
