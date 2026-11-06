"""Write/validate the competition CSV using original-image pixel xywh boxes."""

import argparse
import csv
import math
from pathlib import Path

from .evaluate import index_images, normalize_predictions, read_json


def read_test_images(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "file_name", "clip", "frame", "width", "height"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Test manifest must contain {sorted(required)}")
        images = [{"id": row["image_id"], "file_name": row["file_name"],
                   "video": row["clip"], "frame_id": int(row["frame"]),
                   "width": int(row["width"]), "height": int(row["height"])} for row in reader]
    index_images(images)
    return images


def write_submission(images, predictions, output):
    """Input: COCO records with test string image ids and categories 1..8."""
    predictions, counts = normalize_predictions(predictions, images)
    grouped = {image["id"]: [] for image in images}
    for prediction in predictions:
        values = [prediction["category_id"] - 1, prediction["score"], *prediction["bbox"]]
        grouped[prediction["image_id"]].extend([str(values[0]), *(f"{v:.12g}" for v in values[1:])])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows((image["id"], " ".join(grouped[image["id"]])) for image in images)
    validate_submission(images, output)
    return counts


def validate_submission(images, path):
    """Reject malformed rows, missing/repeated ids, bad boxes and >500 detections."""
    index = index_images(images)
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "PredictionString"]:
            raise ValueError("Submission header must be image_id,PredictionString")
        rows = list(reader)
    if [r["image_id"] for r in rows] != list(index):
        raise ValueError("Submission must contain all manifest ids once, in manifest order")
    detections = nonempty = 0
    for row in rows:
        if None in row or row["PredictionString"] is None:
            raise ValueError("Submission row must have exactly two columns")
        tokens = row["PredictionString"].split()
        if len(tokens) % 6 or len(tokens) > 500 * 6:
            raise ValueError("PredictionString must contain at most 500 groups of six values")
        image = index[row["image_id"]]
        nonempty += bool(tokens)
        for offset in range(0, len(tokens), 6):
            cls, score, x, y, width, height = tokens[offset:offset + 6]
            if cls not in {str(i) for i in range(8)}:
                raise ValueError("Submission classes must be integers 0..7")
            score, x, y, width, height = map(float, (score, x, y, width, height))
            if not all(math.isfinite(v) for v in (score, x, y, width, height)):
                raise ValueError("Nonfinite submission value")
            if not 0 <= score <= 1 or width <= 0 or height <= 0:
                raise ValueError("Invalid score or box size")
            if x < 0 or y < 0 or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
                raise ValueError("Submission boxes must be clipped to original image bounds")
            detections += 1
    return {"images": len(rows), "images_with_predictions": nonempty, "detections": detections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-images", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pred", help="COCO JSON; string image ids, category ids 1..8")
    action.add_argument("--validate", help="Existing submission CSV to check")
    parser.add_argument("--output")
    args = parser.parse_args()
    images = read_test_images(args.test_images)
    if args.validate:
        print(validate_submission(images, args.validate))
    elif args.output:
        print(write_submission(images, read_json(args.pred), args.output))
    else:
        parser.error("--output is required with --pred")


if __name__ == "__main__":
    main()
