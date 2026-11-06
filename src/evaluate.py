"""COCO bbox evaluation with a global per-image detection cap.

Sources: github.com/cocodataset/cocoapi; mixaill76.github.io/faster_coco_eval
"""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path

from .convert import DEFAULT_CLASSES

CLASS_NAMES = tuple(DEFAULT_CLASSES)
CATEGORY_IDS = set(range(1, 9))
MAX_DETECTIONS = 500


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def index_images(images):
    """Require unique image ids and valid original dimensions."""
    index = {}
    for image in images:
        image_id = image["id"]
        if isinstance(image_id, bool) or not isinstance(image_id, (str, int)):
            raise ValueError(f"Invalid image id: {image_id!r}")
        if image_id in index:
            raise ValueError(f"Duplicate image id: {image_id!r}")
        for key in ("width", "height"):
            value = image[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid {key} for image {image_id!r}: {value!r}")
        index[image_id] = image
    if not index:
        raise ValueError("Image manifest is empty")
    return index


def image_key(path):
    """Use clip/frame, never the frame basename shared by multiple videos."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def normalize_predictions(predictions, images, max_detections=MAX_DETECTIONS):
    """Validate COCO predictions, clip to original images, then global top-k.

    Completely out-of-image boxes are dropped after clipping. Invalid values,
    unknown image/category ids and nonpositive input box sizes raise errors.
    Score ties preserve input order. No confidence threshold is introduced.
    """
    if not isinstance(max_detections, int) or not 1 <= max_detections <= MAX_DETECTIONS:
        raise ValueError("max_detections must be an integer in 1..500")
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON list of COCO detection records")
    index = index_images(images)
    grouped = defaultdict(list)
    clipped = outside = 0
    for i, prediction in enumerate(predictions):
        image_id = prediction["image_id"]
        if isinstance(image_id, bool) or image_id not in index:
            raise ValueError(f"Prediction {i}: unknown image_id {image_id!r}")
        category = prediction["category_id"]
        if type(category) is not int or category not in CATEGORY_IDS:
            raise ValueError(f"Prediction {i}: COCO category_id must be in 1..8")
        score = prediction["score"]
        bbox = prediction["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Prediction {i}: bbox must be [x, y, width, height]")
        values = [score, *bbox]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Prediction {i}: score and box coordinates must be finite numbers")
        if not 0 <= score <= 1:
            raise ValueError(f"Prediction {i}: score must be in [0, 1]")
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError(f"Prediction {i}: width and height must be positive")
        image = index[image_id]
        x1, y1 = max(0.0, min(x, image["width"])), max(0.0, min(y, image["height"]))
        x2 = max(0.0, min(x + width, image["width"]))
        y2 = max(0.0, min(y + height, image["height"]))
        if x2 <= x1 or y2 <= y1:
            outside += 1
            continue
        box = [x1, y1, x2 - x1, y2 - y1]
        clipped += box != list(bbox)
        grouped[image_id].append({"image_id": image_id, "category_id": category,
                                  "score": float(score), "bbox": box})
    output = []
    truncated = 0
    for image_id in index:
        detections = sorted(grouped[image_id], key=lambda p: -p["score"])
        truncated += max(0, len(detections) - max_detections)
        output.extend(detections[:max_detections])
    return output, {"input_detections": len(predictions), "kept_detections": len(output),
                    "clipped_detections": clipped, "outside_image_dropped": outside,
                    "over_cap_dropped": truncated, "images": len(index),
                    "images_with_predictions": sum(bool(values) for values in grouped.values())}


def oracle_predictions(ground_truth):
    """GT-as-predictions sanity check, excluding crowd/ignored annotations."""
    return [{"image_id": a["image_id"], "category_id": a["category_id"],
             "bbox": list(a["bbox"]), "score": 1.0}
            for a in ground_truth["annotations"]
            if not a.get("iscrowd", 0) and not a.get("ignore", 0)]


def subset_ground_truth(ground_truth, image_ids):
    """Subset without rewriting or replicating ignore annotations."""
    wanted = set(image_ids)
    known = set(index_images(ground_truth["images"]))
    if not wanted or not wanted <= known:
        raise ValueError("Requested GT image ids must be a nonempty subset")
    return {**ground_truth,
            "images": [i for i in ground_truth["images"] if i["id"] in wanted],
            "annotations": [a for a in ground_truth["annotations"] if a["image_id"] in wanted]}


def evaluate_predictions(ground_truth, predictions, *, backend="faster", max_detections=500):
    """Return AP fractions (0..1); unavailable area/class metrics are None.

    Read the precision tensor at the configured maxDets rather than relying on
    pycocotools' default summary AP, which assumes maxDets=100 in some versions.
    Crowd handling is the backend's standard category-aware COCO behavior.
    """
    import numpy as np

    if backend == "faster":
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    elif backend == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    else:
        raise ValueError(f"Unknown backend: {backend}")
    category_ids = [c["id"] for c in ground_truth["categories"]]
    if len(category_ids) != 8 or set(category_ids) != CATEGORY_IDS:
        raise ValueError("GT must define exactly the eight COCO categories 1..8")
    index = index_images(ground_truth["images"])
    annotation_ids = set()
    for annotation in ground_truth["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError(f"Duplicate annotation id: {annotation['id']}")
        annotation_ids.add(annotation["id"])
        if annotation["image_id"] not in index or annotation["category_id"] not in CATEGORY_IDS:
            raise ValueError("GT annotation references an unknown image or category")
    cleaned, counts = normalize_predictions(predictions, ground_truth["images"], max_detections)
    # Backends mutate annotation dictionaries while preparing their indexes.
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(ground_truth)
        gt.createIndex()
        if cleaned:
            dt = gt.loadRes(cleaned)
        else:
            dt = COCO()
            dt.dataset = {"images": deepcopy(ground_truth["images"]),
                          "categories": deepcopy(ground_truth["categories"]), "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = list(index)
        evaluator.params.catIds = sorted(CATEGORY_IDS)
        evaluator.params.iouThrs = np.linspace(0.5, 0.95, 10)
        evaluator.params.maxDets = sorted(set([1, 10, max_detections]))
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDets
    max_index = list(evaluator.params.maxDets).index(max_detections)

    def ap(area="all", iou=None, category=None):
        tensor = precision[..., evaluator.params.areaRngLbl.index(area), max_index]
        if iou is not None:
            tensor = tensor[np.isclose(evaluator.params.iouThrs, iou)]
        if category is not None:
            tensor = tensor[:, :, list(evaluator.params.catIds).index(category)]
        valid = tensor[tensor > -1]
        return float(valid.mean()) if valid.size else None

    metrics = {"mAP": ap(), "AP50": ap(iou=0.5), "AP75": ap(iou=0.75),
               "AP_small": ap("small"), "AP_medium": ap("medium"), "AP_large": ap("large")}
    per_class = {c["name"]: {"mAP": ap(category=c["id"]), "AP50": ap(iou=0.5, category=c["id"]),
                              "AP_small": ap("small", category=c["id"])}
                 for c in ground_truth["categories"]}
    return {"metric_status": "COCO diagnostics; competition scorer parity unconfirmed",
            "backend": backend, "iou_thresholds": evaluator.params.iouThrs.tolist(),
            "global_max_detections": max_detections, "metrics": metrics,
            "per_class": per_class, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred", help="COCO JSON predictions, category ids 1..8")
    source.add_argument("--oracle", action="store_true", help="Use GT as predictions (sanity check only)")
    parser.add_argument("--backend", choices=["faster", "pycocotools"], default="faster")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ground_truth = read_json(args.gt)
    predictions = oracle_predictions(ground_truth) if args.oracle else read_json(args.pred)
    report = evaluate_predictions(ground_truth, predictions, backend=args.backend)
    report.update({"ground_truth": args.gt, "predictions": args.pred, "oracle_check": args.oracle})
    write_json(args.output, report)
    print(report["metric_status"])
    print(" | ".join(f"{k}={v:.4f}" if v is not None else f"{k}=n/a" for k, v in report["metrics"].items()))


if __name__ == "__main__":
    main()
