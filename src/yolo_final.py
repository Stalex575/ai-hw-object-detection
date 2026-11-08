"""Final YOLO model: train on ALL training clips (same frames as the final Faster R-CNN), score on the untouched
official val/ split with the shared evaluator, and write a test submission.

Unlike the fold-0 YOLO runs, this keeps the Ultralytics default augmentations (mosaic, HSV, translate, scale)
and uses plain SGD; the fold-0 recipe had them switched off and peaked early.

    python -m src.yolo_final --run-name yolo26s_final_all_step2_e12_b4_s42_v1
"""

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def predict_hflip(model, images, root, output, imgsz, batch=4):
    """Test-time augmentation pass: predict on mirrored frames, mirror the boxes back (same settings as yolo.predict)."""
    import cv2
    from .evaluate import normalize_predictions, write_json

    predictions = []
    for start in range(0, len(images), batch):
        group = images[start:start + batch]
        arrays = []
        for image in group:
            array = cv2.imread(str(Path(root) / image["file_name"]))
            if array is None or array.shape[:2] != (image["height"], image["width"]):
                raise ValueError(f"Unreadable image or size mismatch: {image['file_name']}")
            arrays.append(cv2.flip(array, 1))
        results = model.predict(arrays, batch=batch, imgsz=imgsz, device=0, quantize=16, end2end=True,
                                conf=0.001, iou=0.7, max_det=500, rect=False, augment=False, verbose=False)
        for image, result in zip(group, results):
            width = image["width"]
            for (x1, y1, x2, y2), score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                                    result.boxes.cls.tolist()):
                x1, x2 = width - x2, width - x1
                if x2 > x1 and y2 > y1:
                    predictions.append({"image_id": image["id"], "category_id": int(cls) + 1, "score": score,
                                        "bbox": [x1, y1, x2 - x1, y2 - y1]})
    predictions, _ = normalize_predictions(predictions, images)
    write_json(output, predictions)
    return predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default=str(REPO / "notebooks/yolo26s.pt"))
    p.add_argument("--train-list", default=str(REPO / "prep/yolo/train_step2_masked.txt"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--close-mosaic", type=int, default=3, help="last N epochs without mosaic")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="continue from <out>/train/weights/last.pt")
    p.add_argument("--skip-train", action="store_true", help="only predict/evaluate with the existing last.pt")
    p.add_argument("--hflip-only", action="store_true",
                   help="no training: ensemble inputs (plain + mirrored) on all official val frames and test")
    p.add_argument("--weights", default="last.pt", help="checkpoint under train/weights/ for --hflip-only")
    args = p.parse_args()

    out = REPO / "outputs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    for name in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        directory = REPO / "outputs" / name.lower()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(directory))

    from ultralytics import YOLO, settings
    from . import evaluate, submit, yolo

    settings.update({"wandb": False})
    data_yaml = out / "data.yaml"
    names = {i: n for i, n in enumerate(evaluate.CLASS_NAMES)}
    data_yaml.write_text(json.dumps({"path": str(REPO / "prep/yolo"), "train": args.train_list,
                                     "val": str(REPO / "prep/yolo/officialval_step5.txt"), "names": names}))
    weights = out / "train/weights/last.pt"
    if args.hflip_only:
        # Ensemble inputs: plain + mirrored passes on ALL official val frames (consecutive, for the temporal
        # rescoring) and on test, for the checkpoint given by --weights. Files are tagged with its name.
        model = YOLO(str(out / "train/weights" / args.weights))
        name = Path(args.weights).stem
        gt = evaluate.read_json(REPO / "prep/coco/val.json")
        test_images = submit.read_test_images(REPO / "test_images.csv")
        for tag, run in (("", yolo.predict), ("_hflip", predict_hflip)):
            predictions = run(model, gt["images"], REPO / "val", out / f"valall_{name}{tag}.json",
                              imgsz=args.imgsz, batch=4)
            report = evaluate.evaluate_predictions(gt, predictions)
            evaluate.write_json(out / f"metrics_valall_{name}{tag}.json", report)
            print(f"all official val frames, {name}{tag}", json.dumps(report["metrics"]), flush=True)
            step5 = {i["file_name"] for i in evaluate.read_json(REPO / "prep/coco/val_step5.json")["images"]}
            ids = {i["id"] for i in gt["images"] if i["file_name"] in step5}
            sub = evaluate.subset_ground_truth(gt, ids)
            report5 = evaluate.evaluate_predictions(sub, [d for d in predictions if d["image_id"] in ids])
            print(f"official val step 5, {name}{tag}", json.dumps(report5["metrics"]), flush=True)
            run(model, test_images, REPO, out / f"test_{name}{tag}.json", imgsz=args.imgsz, batch=4)
        return
    started = time.perf_counter()
    if not args.skip_train:
        if args.resume and weights.exists():
            YOLO(str(weights)).train(resume=True)
        else:
            # Everything not listed is the Ultralytics default (mosaic 1.0, hsv 0.015/0.7/0.4, translate 0.1,
            # scale 0.5, fliplr 0.5), i.e. the augmentations the fold-0 recipe had switched off.
            YOLO(args.model).train(
                data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0,
                workers=args.workers, seed=args.seed, optimizer="SGD", lr0=args.lr0, momentum=0.937,
                warmup_epochs=1, close_mosaic=args.close_mosaic, max_det=500, amp=True, cache=False,
                project=str(out), name="train", exist_ok=True, save_period=1, plots=True, verbose=False)
    hours = (time.perf_counter() - started) / 3600

    # Fixed-length protocol: report the last epoch, like every other run.
    model = YOLO(str(weights))
    gt = evaluate.read_json(REPO / "prep/coco/val_step5.json")
    predictions = yolo.predict(model, gt["images"], REPO / "val", out / "val_predictions.json",
                               imgsz=args.imgsz, batch=4)
    report = evaluate.evaluate_predictions(gt, predictions)
    evaluate.write_json(out / "metrics.json", report)
    print("official val metrics", json.dumps(report["metrics"]), flush=True)

    test_images = submit.read_test_images(REPO / "test_images.csv")
    test_predictions = yolo.predict(model, test_images, REPO, out / "test_predictions.json",
                                    imgsz=args.imgsz, batch=4)
    submit.write_submission(test_images, test_predictions, out / "submission.csv")
    check = submit.validate_submission(test_images, out / "submission.csv")
    evaluate.write_json(out / "submission_check.json", check)
    print("submission", json.dumps(check), flush=True)
    evaluate.write_json(out / "summary.json", {"run": args.run_name, "official_val": report["metrics"],
                                                "train_hours_this_call": hours, "args": vars(args)})


if __name__ == "__main__":
    main()
