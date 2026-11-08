"""Temporal rescoring for video frames (a simple Seq-NMS-style idea).

The test set is consecutive frames of 48 clips. A real object appears at almost the same place in the neighbouring
frames; a false positive tends to flicker. For every detection we look at the K previous and K next frames of the
same clip, take the best-scoring detection of the same class that overlaps it (IoU >= thr, 0 if none), and blend:

    new_score = (1 - alpha) * score + alpha * mean(neighbour matches)

Boxes are not moved; only the ranking changes, which is what mAP measures.

    # grid on the official val/ split (all frames), then apply the chosen setting to test
    python -m src.temporal --val outputs/ens/valall_predictions.json --gt prep/coco/val.json
    python -m src.temporal --val ... --test outputs/ens/test_predictions.json --k 2 --alpha 0.4 --thr 0.3 --out outputs/ens_t
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import submit
from .evaluate import evaluate_predictions, normalize_predictions, read_json, write_json

REPO = Path(__file__).resolve().parents[1]


def iou_matrix(a, b):
    """a: (n, 4), b: (m, 4) xyxy -> (n, m)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def index(predictions, images):
    """Per (image, class): xyxy boxes, scores and the prediction indices; per clip: image ids in frame order."""
    per = defaultdict(list)
    for i, d in enumerate(predictions):
        per[(d["image_id"], d["category_id"])].append(i)
    arrays = {}
    for key, idx in per.items():
        b = np.asarray([predictions[i]["bbox"] for i in idx], float)
        arrays[key] = (np.c_[b[:, :2], b[:, :2] + b[:, 2:]], np.asarray([predictions[i]["score"] for i in idx]), idx)
    clips = defaultdict(list)
    for image in images:
        clips[image["video"]].append((image["frame_id"], image["id"]))
    order = {v: [i for _, i in sorted(frames)] for v, frames in clips.items()}
    return arrays, order


def rescore(predictions, images, k=2, alpha=0.4, thr=0.3, prepared=None):
    arrays, order = prepared or index(predictions, images)
    categories = {c for _, c in arrays}
    new = [dict(d) for d in predictions]
    for frames in order.values():
        for position, image_id in enumerate(frames):
            neighbours = [frames[j] for j in range(position - k, position + k + 1)
                          if j != position and 0 <= j < len(frames)]
            if not neighbours:
                continue
            for c in categories:
                own = arrays.get((image_id, c))
                if own is None:
                    continue
                boxes, scores, idx = own
                support = np.zeros(len(idx))
                for n in neighbours:
                    other = arrays.get((n, c))
                    if other is None:
                        continue  # no detection of this class there: contributes 0
                    overlap = iou_matrix(boxes, other[0])
                    support += np.where(overlap >= thr, other[1][None, :], 0).max(1)
                support /= len(neighbours)
                for i, s, m in zip(idx, scores, support):
                    new[i]["score"] = float((1 - alpha) * s + alpha * m)
    return new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", required=True)
    p.add_argument("--gt", default=str(REPO / "prep/coco/val.json"))
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    p.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5])
    p.add_argument("--test", help="test predictions to rescore with the (single) given setting")
    p.add_argument("--out")
    args = p.parse_args(argv)
    gt = read_json(args.gt)
    val = read_json(args.val)
    base = evaluate_predictions(gt, val)["metrics"]
    print(json.dumps({"setting": "none", **{m: round(v, 4) for m, v in base.items()}}), flush=True)
    prepared = index(val, gt["images"])
    results = []
    for k, alpha, thr in itertools.product(args.k, args.alpha, args.thr):
        metrics = evaluate_predictions(gt, rescore(val, gt["images"], k, alpha, thr, prepared))["metrics"]
        results.append({"k": k, "alpha": alpha, "thr": thr, **{m: round(v, 4) for m, v in metrics.items()}})
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "temporal_grid.json", {"base": base, "grid": results})
        if args.test:
            if len(results) != 1:
                raise ValueError("Give exactly one --k/--alpha/--thr for the test pass")
            images = submit.read_test_images(REPO / "test_images.csv")
            test = rescore(read_json(args.test), images, args.k[0], args.alpha[0], args.thr[0])
            test, _ = normalize_predictions(test, images)
            write_json(out / "test_predictions.json", test)
            submit.write_submission(images, test, out / "submission.csv")
            check = submit.validate_submission(images, out / "submission.csv")
            write_json(out / "submission_check.json", check)
            print("submission", json.dumps(check), flush=True)


if __name__ == "__main__":
    main()
