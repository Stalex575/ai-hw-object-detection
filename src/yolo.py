"""YOLO fold preparation, prediction export and batch-one timing."""

from pathlib import Path
import statistics
import time

import yaml

from . import convert
from .evaluate import CLASS_NAMES, normalize_predictions, read_json, write_json


def image_key(path):
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def prepare_fold(data, prep, folds, fold=0, step=5):
    data, prep, folds = Path(data).resolve(), Path(prep).resolve(), Path(folds).resolve()
    gt_path = prep / "coco" / f"train_step{step}_fold{fold}_val.json"
    source_yaml = prep / "yolo" / f"train_step{step}_masked_fold{fold}.yaml"
    if not gt_path.exists():
        convert.main(["coco", "--split-dir", str(data / "train"), "--out",
                      str(prep / "coco" / f"train_step{step}.json"),
                      "--folds", str(folds), "--frame-step", str(step)])
    if not source_yaml.exists():
        convert.main(["yolo", "--split-dir", str(data / "train"), "--out", str(prep / "yolo"),
                      "--folds", str(folds), "--frame-step", str(step)])
    config = yaml.safe_load(source_yaml.read_text())
    train_list = source_yaml.parent / config["train"]
    train_paths = [Path(p) for p in train_list.read_text().splitlines() if p.strip()]
    truth = read_json(gt_path)
    val_paths = [data / "train" / image["file_name"] for image in truth["images"]]
    fold_map = convert.read_folds(folds)
    for role, paths in (("train", train_paths), ("val", val_paths)):
        keys = [image_key(p) for p in paths]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"Empty/duplicate images in {role}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if (fold_map[path.parent.name] == fold) != (role == "val"):
                raise ValueError(f"Fold assignment mismatch: {path}")
            prefix, relative = str(path).rsplit("/images/", 1)
            label = Path(prefix, "labels", relative).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
    train_truth = read_json(gt_path.with_name(f"train_step{step}_fold{fold}_train.json"))
    if {image_key(i["file_name"]) for i in train_truth["images"]} != {image_key(p) for p in train_paths}:
        raise ValueError("Prepared training list and fold GT disagree")
    holdout = read_json(data / "val/annotations_coco.json")
    held_out = {convert.image_clip(i) for i in holdout["images"]}
    if held_out & {p.parent.name for p in train_paths + val_paths}:
        raise ValueError("Official holdout clips overlap the tuning fold")
    val_list = prep / "yolo" / f"fold{fold}_val_original_step{step}.txt"
    val_list.write_text("".join(f"{p}\n" for p in val_paths))
    run_yaml = source_yaml.with_name(f"fold{fold}_step{step}.yaml")
    run_yaml.write_text(yaml.safe_dump({"path": str(source_yaml.parent), "train": str(train_list),
                                       "val": str(val_list), "names": list(CLASS_NAMES)}, sort_keys=False))
    return run_yaml, gt_path, {"train_images": len(train_paths), "val_images": len(val_paths),
                               "fold": fold, "frame_step": step}


def predict(model, images, root, output, *, imgsz=640, batch=8, device=0, half=True, end2end=True):
    names = model.names
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if list(names) != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order differs from the competition")
    predictions, seen = [], set()
    for start in range(0, len(images), batch):
        group = images[start:start + batch]
        paths = [Path(root) / image["file_name"] for image in group]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        expected = {image_key(p): image for p, image in zip(paths, group)}
        if len(expected) != len(group):
            raise ValueError("Duplicate clip/frame keys")
        results = model.predict([str(p) for p in paths], stream=True, batch=batch,
                                imgsz=imgsz, device=device, quantize=16 if half else 32, end2end=end2end,
                                conf=0.001, iou=0.7, max_det=500, rect=False, augment=False, verbose=False)
        for result in results:
            key = image_key(result.path)
            image = expected[key]
            if key in seen or tuple(result.orig_shape) != (image["height"], image["width"]):
                raise ValueError(f"Duplicate image or original-size mismatch: {key}")
            seen.add(key)
            if result.boxes is None:
                continue
            for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                if cls != int(cls) or not 0 <= cls < 8:
                    raise ValueError(f"Invalid model class: {cls}")
                x1, y1, x2, y2 = box
                predictions.append({"image_id": image["id"], "category_id": int(cls) + 1,
                                    "score": score, "bbox": [x1, y1, x2 - x1, y2 - y1]})
    if len(seen) != len(images):
        raise ValueError(f"Incomplete inference: {len(seen)}/{len(images)} images")
    predictions, counts = normalize_predictions(predictions, images)
    write_json(output, predictions)
    write_json(str(output) + ".manifest.json", {**counts, "completed_images": len(seen), "end2end": end2end})
    return predictions


def benchmark(model, paths, *, imgsz=640, device=0, half=True, end2end=True, warmup=20, repeats=100):
    import cv2
    import torch

    if not paths or warmup < 1 or repeats < 1:
        raise ValueError("Positive sample, warmup and repeat counts required")
    if str(device).isdigit():
        device = int(device)
    cuda = str(device) not in {"cpu", "mps"}

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)
        elif str(device) == "mps":
            torch.mps.synchronize()

    durations = []
    for i in range(warmup + repeats):
        synchronize()
        start = time.perf_counter()
        image = cv2.imread(str(paths[i % len(paths)]))
        if image is None:
            raise ValueError("Image decoding failed")
        result = model.predict(image, imgsz=imgsz, device=device, quantize=16 if half else 32, end2end=end2end,
                               conf=0.001, iou=0.7, max_det=500, rect=False, augment=False, verbose=False)
        result[0].boxes.cpu()
        synchronize()
        if i >= warmup:
            durations.append(time.perf_counter() - start)
    return {"fps": repeats / sum(durations), "mean_ms": statistics.mean(durations) * 1000,
            "batch": 1, "warmup": warmup, "repeats": repeats, "imgsz": imgsz, "half": half,
            "end2end": end2end, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "scope": "image decode, preprocessing, inference, post-processing and CPU box transfer"}
