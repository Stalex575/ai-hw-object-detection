"""TIDE-style error breakdown (Bolya et al., ECCV 2020) of a COCO prediction JSON at IoU 0.5.

Every detection that is not a true positive (greedy, score-ordered, per-class matching at t_f = 0.5,
as in COCO) and does not fall in a same-class crowd/ignore region gets one error type:

    Dup   IoU >= t_f with a same-class GT that a higher-scoring detection already matched
    Cls   IoU >= t_f with a GT of another class
    Loc   t_b <= IoU < t_f with a same-class GT (t_b = 0.1)
    Both  t_b <= IoU < t_f with a GT of another class only
    Bkg   IoU < t_b with every GT
    Miss  GT boxes left unmatched that no Cls/Loc detection points at

Impact of an error type = AP50 after fixing only that type minus the baseline AP50, with the shared evaluator.
Fixes, following TIDE: Cls -> relabel to the GT's class, Loc -> snap the box to the GT
(both suppressed instead if that GT is already taken); Dup / Both / Bkg -> suppress; Miss -> remove those GT.
Detections below --min-score are dropped first (default 0: keep all; 0.01 cost FRCNN ~1 AP50 point).

    python -m src.tide outputs/frcnn_base_fold0_1280_e12_b2_s42_v1/val_predictions.json --out experiments/tide_frcnn
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .evaluate import evaluate_predictions, read_json, write_json

REPO = Path(__file__).resolve().parents[1]
TYPES = ("Cls", "Loc", "Both", "Dup", "Bkg", "Miss")


def xyxy(boxes):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], 1)


def iou(a, b, crowd=False):
    """IoU of a (N,4) vs b (M,4) xyxy; with crowd=True the denominator is the area of a (COCO crowd rule)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    if crowd:
        return inter / np.maximum(area_a[:, None], 1e-9)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def classify(gt, predictions, t_f=0.5, t_b=0.1):
    gts, crowds = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        (crowds if a.get("iscrowd") or a.get("ignore") else gts)[a["image_id"]].append(a)
    dets = defaultdict(list)
    for i, d in enumerate(predictions):
        dets[d["image_id"]].append(i)
    kind = {}            # detection index -> "TP" | "ignored" | error type
    target = {}          # detection index -> GT annotation id it points at (Cls / Loc / Dup)
    matched = set()      # GT ids matched by a TP
    for image in gt["images"]:
        g = gts[image["id"]]
        order = sorted(dets[image["id"]], key=lambda i: -predictions[i]["score"])
        if not order:
            continue
        d_box = xyxy([predictions[i]["bbox"] for i in order])
        d_cls = np.array([predictions[i]["category_id"] for i in order])
        g_box = xyxy([a["bbox"] for a in g]) if g else np.zeros((0, 4))
        g_cls = np.array([a["category_id"] for a in g])
        ious = iou(d_box, g_box) if g else np.zeros((len(order), 0))
        c = crowds[image["id"]]
        c_ious = iou(d_box, xyxy([a["bbox"] for a in c]), crowd=True) if c else np.zeros((len(order), 0))
        c_cls = np.array([a["category_id"] for a in c])
        taken = np.zeros(len(g), bool)
        for row, i in enumerate(order):
            same = g_cls == d_cls[row]
            cand = np.where(same & ~taken & (ious[row] >= t_f))[0]
            if len(cand):
                j = cand[np.argmax(ious[row, cand])]
                taken[j] = True
                matched.add(g[j]["id"])
                kind[i] = "TP"
                continue
            if c and (c_ious[row, c_cls == d_cls[row]] >= t_f).any():
                kind[i] = "ignored"
                continue
            s = np.where(same, ious[row], 0)
            o = np.where(~same, ious[row], 0)
            if g and s.max() >= t_f:
                kind[i], target[i] = "Dup", g[int(s.argmax())]["id"]
            elif g and o.max() >= t_f:
                kind[i], target[i] = "Cls", g[int(o.argmax())]["id"]
            elif g and s.max() >= t_b:
                kind[i], target[i] = "Loc", g[int(s.argmax())]["id"]
            elif g and o.max() >= t_b:
                kind[i] = "Both"
            else:
                kind[i] = "Bkg"
    pointed = {target[i] for i in target if kind[i] in ("Cls", "Loc")}
    missed = [a["id"] for v in gts.values() for a in v if a["id"] not in matched and a["id"] not in pointed]
    return kind, target, matched, missed


def fixed(gt, predictions, kind, target, matched, missed, error):
    """Return (gt, predictions) with only `error` fixed."""
    if error == "Miss":
        drop = set(missed)
        return {**gt, "annotations": [a for a in gt["annotations"] if a["id"] not in drop]}, predictions
    by_id = {a["id"]: a for a in gt["annotations"]}
    used, out = set(matched), []
    for i in sorted(range(len(predictions)), key=lambda i: -predictions[i]["score"]):
        d, k = predictions[i], kind.get(i)
        if k != error:
            out.append(d)
        elif k in ("Cls", "Loc"):
            a = by_id[target[i]]
            if a["id"] in used:
                continue  # would be a duplicate after the fix: suppress
            used.add(a["id"])
            if k == "Cls":
                out.append({**d, "category_id": a["category_id"]})  # relabel, keep the box
            else:
                out.append({**d, "bbox": list(a["bbox"])})  # snap the box, keep the class
        # Dup / Both / Bkg: suppressed
    return gt, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions")
    p.add_argument("--gt", default=str(REPO / "prep/coco/train_step5_fold0_val.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--examples", type=int, default=3, help="failure-case frames to draw")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = read_json(args.gt)
    predictions = [d for d in read_json(args.predictions) if d["score"] >= args.min_score]
    kind, target, matched, missed = classify(gt, predictions)
    base = evaluate_predictions(gt, predictions)["metrics"]["AP50"]
    counts = Counter(kind.values())
    counts["Miss"] = len(missed)
    weighted = Counter()
    for i, k in kind.items():
        weighted[k] += predictions[i]["score"]
    names = {c["id"]: c["name"] for c in gt["categories"]}
    gt_class = {a["id"]: a["category_id"] for a in gt["annotations"]}
    confusion = Counter()
    for i, k in kind.items():
        if k == "Cls":
            confusion[f"{names[predictions[i]['category_id']]} -> GT {names[gt_class[target[i]]]}"] += predictions[i]["score"]
    report = {"cls_confusion_score_weighted": {k: round(v, 1) for k, v in confusion.most_common(12)},
              "predictions": args.predictions, "min_score": args.min_score, "t_f": 0.5, "t_b": 0.1,
              "AP50": base, "counts": dict(counts), "score_weighted_counts": {k: round(v, 1) for k, v in weighted.items()},
              "dAP50": {}}
    for error in TYPES:
        g, d = fixed(gt, predictions, kind, target, matched, missed, error)
        report["dAP50"][error] = evaluate_predictions(g, d)["metrics"]["AP50"] - base
        print(error, round(report["dAP50"][error] * 100, 2), "AP50 points", flush=True)
    # All false positives at once / all false negatives at once (TIDE's "special" errors).
    fp = [d for i, d in enumerate(predictions) if kind.get(i) in ("TP", "ignored")]
    report["dAP50"]["FP_all"] = evaluate_predictions(gt, fp)["metrics"]["AP50"] - base
    tp_gt = matched
    g = {**gt, "annotations": [a for a in gt["annotations"]
                               if a.get("iscrowd") or a.get("ignore") or a["id"] in tp_gt]}
    report["dAP50"]["FN_all"] = evaluate_predictions(g, predictions)["metrics"]["AP50"] - base
    write_json(out / "tide.json", report)
    plot(report, out / "tide.png")
    draw_examples(gt, predictions, kind, missed, out, args.examples)
    print(report)


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(TYPES) + ["FP_all", "FN_all"]
    values = [report["dAP50"][n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(names, values, color=["#2a78d6"] * len(TYPES) + ["#52514e"] * 2)
    for x, v in enumerate(values):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("AP50 gain if fixed (points)")
    ax.set_title(f"TIDE-style errors at IoU 0.5 (AP50 = {report['AP50'] * 100:.1f})")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def draw_examples(gt, predictions, kind, missed, out, n, min_score=0.3):
    """Frames with the most confident errors: GT missed (yellow), Bkg (red), Loc (orange), Cls (magenta),
    Dup (cyan), TP (green); detections drawn only above min_score."""
    if n <= 0:
        return
    from PIL import Image, ImageDraw
    colors = {"TP": "lime", "Bkg": "red", "Loc": "orange", "Cls": "magenta", "Both": "purple", "Dup": "cyan"}
    names = {c["id"]: c["name"] for c in gt["categories"]}
    missed = set(missed)
    by_image = defaultdict(list)
    for i, d in enumerate(predictions):
        by_image[d["image_id"]].append(i)
    miss_by_image = Counter(a["image_id"] for a in gt["annotations"] if a["id"] in missed)
    score = {img["id"]: miss_by_image[img["id"]] + sum(1 for i in by_image[img["id"]]
                                                       if predictions[i]["score"] >= min_score
                                                       and kind.get(i) not in ("TP", "ignored", None))
             for img in gt["images"]}
    clips, chosen = set(), []
    for image_id in sorted(score, key=lambda k: -score[k]):
        info = next(i for i in gt["images"] if i["id"] == image_id)
        clip = Path(info["file_name"]).parent.name
        if clip in clips:
            continue
        clips.add(clip)
        chosen.append(info)
        if len(chosen) == n:
            break
    for rank, info in enumerate(chosen, 1):
        picture = Image.open(REPO / "train" / info["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(picture)
        for a in gt["annotations"]:
            if a["image_id"] == info["id"] and a["id"] in missed:
                x, y, w, h = a["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline="yellow", width=3)
        for i in by_image[info["id"]]:
            d = predictions[i]
            k = kind.get(i)
            if d["score"] < min_score or k not in colors:
                continue
            x, y, w, h = d["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=colors[k], width=2)
            if k != "TP":
                draw.text((x, max(0, y - 11)), f"{k} {names[d['category_id']]} {d['score']:.2f}", fill=colors[k])
        draw.text((10, 10), f"{info['file_name']} | yellow: missed GT, green: TP, red: Bkg, orange: Loc, "
                  f"magenta: Cls, cyan: Dup (score >= {min_score})", fill="white", stroke_width=2, stroke_fill="black")
        picture.save(out / f"failure_{rank}.jpg", quality=88)


if __name__ == "__main__":
    main()
