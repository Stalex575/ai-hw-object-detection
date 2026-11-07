"""Re-evaluate a finished Ultralytics run on train/weights/last.pt (protocol rule: last epoch, not best.pt).

Writes <run>/last/{val_predictions.json, metrics.json, test_predictions.json, submission.csv,
submission_check.json, speed.json, experiment.csv}. The original best.pt outputs in <run>/ stay untouched.

    python tools/queue/reeval_last.py outputs/yolo26s_base_fold0_1280_e25_s42_v1 --model yolo --imgsz 1280
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import evaluate, submit, yolo  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--model", choices=["yolo", "rtdetr"], required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--family", default=None)
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: N val images, writes into <run>/last_debug")
    args = p.parse_args()
    from ultralytics import RTDETR, YOLO

    run = Path(args.run).resolve()
    weights = run / "train/weights/last.pt"
    out = run / ("last_debug" if args.limit else "last")
    out.mkdir(exist_ok=True)
    cls = RTDETR if args.model == "rtdetr" else YOLO
    gt = evaluate.read_json(REPO / "prep/coco/train_step5_fold0_val.json")
    if args.limit:
        gt = evaluate.subset_ground_truth(gt, [i["id"] for i in gt["images"][:args.limit]])
    predictions = yolo.predict(cls(weights), gt["images"], REPO / "train", out / "val_predictions.json", imgsz=args.imgsz)
    report = evaluate.evaluate_predictions(gt, predictions)
    report["weights"] = str(weights.relative_to(REPO))
    evaluate.write_json(out / "metrics.json", report)
    print(run.name, report["metrics"], flush=True)
    paths = [REPO / "train" / i["file_name"] for i in gt["images"]]
    sample = [paths[round(i * (len(paths) - 1) / 31)] for i in range(32)]
    speed = yolo.benchmark(cls(weights), sample, imgsz=args.imgsz)
    evaluate.write_json(out / "speed.json", speed)
    if not args.no_test and not args.limit:
        test_images = submit.read_test_images(REPO / "test_images.csv")
        test_predictions = yolo.predict(cls(weights), test_images, REPO, out / "test_predictions.json", imgsz=args.imgsz)
        submit.write_submission(test_images, test_predictions, out / "submission.csv")
        evaluate.write_json(out / "submission_check.json", submit.validate_submission(test_images, out / "submission.csv"))
    old = next(csv.DictReader((run / "experiment.csv").open()))
    row = {**old, "id": run.name + "@last", **report["metrics"], "FPS": speed["fps"], "public_LB": ""}
    with (out / "experiment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
