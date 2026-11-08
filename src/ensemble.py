"""Weighted Boxes Fusion (WBF, Solovyev et al. 2021) of several prediction files, per image and per class.

Every model (or TTA pass) gives a list of COCO detections. For each image and class, boxes from all models are
sorted by score and greedily clustered: a box joins the first cluster whose fused box overlaps it with IoU > thr,
otherwise it starts a new cluster. A cluster's fused box is the score-weighted average of its boxes, and its score
is the weighted mean score, scaled down when fewer models agree (a box only one of two models found keeps half).

    # tune on the official val/ split (prints mAP per setting), then write the test submission
    python -m src.ensemble --val a/val_predictions.json b/val_predictions.json --weights 2 1 --iou 0.55
    python -m src.ensemble --val ... --test a/test_predictions.json b/test_predictions.json --weights 2 1 --out outputs/ens
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import submit
from .evaluate import evaluate_predictions, normalize_predictions, read_json, write_json

REPO = Path(__file__).resolve().parents[1]


def fuse_group(boxes, scores, models, weights, iou_thr):
    """boxes: (n, 4) xyxy; scores, models: (n,). Returns fused (m, 4) boxes and (m,) scores."""
    order = np.argsort(-scores)
    boxes, scores, models = boxes[order], scores[order], models[order]
    w = weights[models]
    fused, members = [], []  # fused xyxy per cluster, member indices per cluster
    for i in range(len(boxes)):
        if fused:
            f = np.asarray(fused)
            ix1 = np.maximum(f[:, 0], boxes[i, 0]); iy1 = np.maximum(f[:, 1], boxes[i, 1])
            ix2 = np.minimum(f[:, 2], boxes[i, 2]); iy2 = np.minimum(f[:, 3], boxes[i, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area_f = (f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1])
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            iou = inter / np.maximum(area_f + area_i - inter, 1e-9)
            j = int(np.argmax(iou))
            if iou[j] > iou_thr:
                members[j].append(i)
                m = members[j]
                cw = scores[m] * w[m]
                fused[j] = (boxes[m] * cw[:, None]).sum(0) / cw.sum()
                continue
        fused.append(boxes[i].copy())
        members.append([i])
    total = weights.sum()
    out_scores = []
    for m in members:
        # weighted mean score of the cluster, times the share of the total model weight that voted for it
        voted = sum(weights[k] for k in set(models[m].tolist()))
        out_scores.append((scores[m] * w[m]).sum() / w[m].sum() * min(voted, total) / total)
    return np.asarray(fused).reshape(-1, 4), np.asarray(out_scores)


def compact(predictions, images):
    """COCO detection list -> numpy arrays (image index, category, score, xyxy); ~10x less RAM than dicts."""
    position = {image["id"]: i for i, image in enumerate(images)}
    n = len(predictions)
    image_index, category = np.empty(n, np.int32), np.empty(n, np.int16)
    score, boxes = np.empty(n, np.float64), np.empty((n, 4), np.float64)
    for i, d in enumerate(predictions):
        image_index[i], category[i], score[i] = position[d["image_id"]], d["category_id"], d["score"]
        x, y, w, h = d["bbox"]
        boxes[i] = (x, y, x + w, y + h)
    return image_index, category, score, boxes


def fuse(prediction_sets, weights, images, iou_thr=0.55, skip_thr=0.0):
    """prediction_sets: COCO detection lists or compact() tuples, one per model/pass."""
    weights = np.asarray(weights, dtype=float)
    parts = [p if isinstance(p, tuple) else compact(p, images) for p in prediction_sets]
    image_index = np.concatenate([p[0] for p in parts]); category = np.concatenate([p[1] for p in parts])
    score = np.concatenate([p[2] for p in parts]); boxes = np.concatenate([p[3] for p in parts])
    model = np.concatenate([np.full(len(p[0]), m) for m, p in enumerate(parts)])
    keep = score >= skip_thr
    image_index, category, score, boxes, model = image_index[keep], category[keep], score[keep], boxes[keep], model[keep]
    key = image_index.astype(np.int64) * 1000 + category
    order = np.argsort(key, kind="stable")
    key, image_index, category, score, boxes, model = (a[order] for a in (key, image_index, category, score, boxes, model))
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    fused = []
    for s, e in zip(starts, ends):
        fb, fs = fuse_group(boxes[s:e], score[s:e], model[s:e], weights, iou_thr)
        image_id, c = images[image_index[s]]["id"], int(category[s])
        for (x1, y1, x2, y2), sc in zip(fb.tolist(), fs.tolist()):
            if x2 > x1 and y2 > y1:
                fused.append({"image_id": image_id, "category_id": c, "score": sc,
                              "bbox": [x1, y1, x2 - x1, y2 - y1]})
    fused, _ = normalize_predictions(fused, images)
    return fused


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", nargs="+", required=True, help="val prediction files, one per model/pass")
    p.add_argument("--gt", default=str(REPO / "prep/coco/val_step5.json"))
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--skip", type=float, default=0.0, help="ignore input boxes below this score")
    p.add_argument("--test", nargs="+", help="test prediction files in the same order as --val")
    p.add_argument("--out", help="output folder for metrics / submission")
    args = p.parse_args(argv)
    weights = args.weights or [1.0] * len(args.val)
    if len(weights) != len(args.val) or (args.test and len(args.test) != len(args.val)):
        raise ValueError("--val, --weights and --test need the same length")
    gt = read_json(args.gt)
    fused = fuse([compact(read_json(f), gt["images"]) for f in args.val], weights, gt["images"], args.iou, args.skip)
    report = evaluate_predictions(gt, fused)
    print(json.dumps({"weights": weights, "iou": args.iou, "skip": args.skip,
                      **{k: round(v, 4) for k, v in report["metrics"].items()}}), flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "metrics.json", report)
        write_json(out / "val_predictions.json", fused)
        info = {"val": args.val, "test": args.test, "weights": weights, "iou": args.iou, "skip": args.skip}
        if args.test:
            images = submit.read_test_images(REPO / "test_images.csv")
            fused_test = fuse([compact(read_json(f), images) for f in args.test], weights, images, args.iou, args.skip)
            write_json(out / "test_predictions.json", fused_test)
            submit.write_submission(images, fused_test, out / "submission.csv")
            info["submission"] = submit.validate_submission(images, out / "submission.csv")
            print("submission", json.dumps(info["submission"]), flush=True)
        write_json(out / "ensemble.json", info)


if __name__ == "__main__":
    main()
