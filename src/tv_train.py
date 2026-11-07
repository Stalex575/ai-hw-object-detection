"""Shared torchvision trainer for Faster R-CNN R50-FPN v2 and RetinaNet R50-FPN v2.

Protocol (PLAN.md): fold-0 clips, every 5th frame, masked training images (same
ignore-region policy as the Ultralytics runs), long side ~1280 px, horizontal flip
+ scale jitter, seed 42, AMP, run length fixed in epochs. Evaluation uses the shared
evaluator on the original (unmasked) fold-0 validation frames.

Example:
    python -m src.tv_train --arch fasterrcnn --epochs 12 --run-name frcnn_base_fold0_1280_e12_s42_v1 --test
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from . import submit
from .evaluate import evaluate_predictions, image_key, normalize_predictions, read_json, write_json

REPO = Path(__file__).resolve().parents[1]
NUM_CLASSES = 9  # 8 classes + background; labels are COCO category ids 1..8


def read_image(path):
    """uint8 RGB CHW tensor -> float in [0, 1]. Normalisation happens inside the model."""
    return decode_image(str(path), mode=ImageReadMode.RGB).float().div_(255)


class TrainSet(Dataset):
    """Masked training frames + boxes from the fold's COCO GT (crowd/ignore boxes dropped)."""

    def __init__(self, image_list, coco_json, hflip=0.5):
        coco = read_json(coco_json)
        by_key = {image_key(i["file_name"]): i for i in coco["images"]}
        boxes = {i["id"]: [] for i in coco["images"]}
        for a in coco["annotations"]:
            if not a.get("iscrowd", 0) and not a.get("ignore", 0):
                boxes[a["image_id"]].append((a["bbox"], a["category_id"]))
        self.items = []
        for line in Path(image_list).read_text().splitlines():
            if not line.strip():
                continue
            info = by_key[image_key(line)]  # KeyError = list and GT disagree
            self.items.append((Path(line), info, boxes[info["id"]]))
        if len(self.items) != len(coco["images"]):
            raise ValueError("Training list and fold GT have different image sets")
        self.hflip = hflip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, info, records = self.items[index]
        image = read_image(path)
        height, width = image.shape[1:]
        if (height, width) != (info["height"], info["width"]):
            raise ValueError(f"Size mismatch: {path}")
        xyxy, labels = [], []
        for (x, y, w, h), category in records:
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                xyxy.append([x1, y1, x2, y2])
                labels.append(category)
        boxes = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        if random.random() < self.hflip:
            image = image.flip(-1)
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], 1)
        return image, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


class ImageSet(Dataset):
    def __init__(self, images, root):
        self.images, self.root = images, Path(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        return read_image(self.root / info["file_name"]), info


def collate(batch):
    return tuple(zip(*batch))


def build_model(arch, train_sizes, test_size, max_size_train, max_size_test):
    from torchvision.models import detection as D

    common = dict(min_size=train_sizes, max_size=max_size_train)
    if arch == "fasterrcnn":
        model = D.fasterrcnn_resnet50_fpn_v2(weights=D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                             box_detections_per_img=500, box_score_thresh=0.001, **common)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
    elif arch == "retinanet":
        model = D.retinanet_resnet50_fpn_v2(weights=D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                                            detections_per_img=500, score_thresh=0.001, **common)
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        model.head.classification_head = RetinaNetClassificationHead(
            256, model.anchor_generator.num_anchors_per_location()[0], NUM_CLASSES,
            norm_layer=lambda c: torch.nn.GroupNorm(32, c))
    else:
        raise ValueError(arch)
    model.size_cfg = {"train": (tuple(train_sizes), max_size_train), "test": ((test_size,), max_size_test)}
    return model


def set_mode(model, train):
    """Switch between scale-jitter training sizes and the fixed test size."""
    sizes, max_size = model.size_cfg["train" if train else "test"]
    model.transform.min_size, model.transform.max_size = sizes, max_size
    model.train(train)


@torch.no_grad()
def predict(model, images, root, device, batch=4, workers=4, amp=True):
    set_mode(model, False)
    loader = DataLoader(ImageSet(images, root), batch_size=batch, num_workers=workers, collate_fn=collate)
    predictions, seen = [], 0
    for batch_images, infos in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outputs = model([i.to(device, non_blocking=True) for i in batch_images])
        for output, info in zip(outputs, infos):
            boxes = output["boxes"].float().cpu().tolist()
            for (x1, y1, x2, y2), score, label in zip(boxes, output["scores"].float().cpu().tolist(),
                                                     output["labels"].cpu().tolist()):
                if x2 > x1 and y2 > y1:
                    predictions.append({"image_id": info["id"], "category_id": int(label),
                                        "score": float(score), "bbox": [x1, y1, x2 - x1, y2 - y1]})
        seen += len(infos)
    if seen != len(images):
        raise ValueError(f"Incomplete inference: {seen}/{len(images)}")
    predictions, _ = normalize_predictions(predictions, images)
    return predictions


@torch.no_grad()
def benchmark(model, paths, device, amp=True, warmup=20, repeats=100):
    """Batch 1, FPS incl. JPEG decode, preprocessing, inference, post-processing and CPU transfer."""
    set_mode(model, False)
    durations = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        image = read_image(paths[i % len(paths)]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            output = model([image])[0]
        output["boxes"].cpu()
        torch.cuda.synchronize(device)
        if i >= warmup:
            durations.append(time.perf_counter() - start)
    return {"fps": repeats / sum(durations), "mean_ms": statistics.mean(durations) * 1000, "batch": 1,
            "warmup": warmup, "repeats": repeats, "half": amp, "gpu": torch.cuda.get_device_name(device),
            "scope": "image decode, preprocessing, inference, post-processing and CPU box transfer"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["fasterrcnn", "retinanet"], default="fasterrcnn")
    p.add_argument("--run-name", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="default: 0.02 * batch / 16 (SGD)")
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--test-size", type=int, default=720, help="short side at test time (16:9 -> 1280 long side)")
    p.add_argument("--max-size", type=int, default=1280, help="long-side cap at test time")
    p.add_argument("--jitter", type=float, default=0.25, help="train short side in [1-j, 1+j] * test-size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=2, help="shared-evaluator fold-0 mAP every N epochs")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--test", action="store_true", help="also predict the test set and write submission.csv")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N train/val images")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=str(REPO))
    p.add_argument("--prep", default=str(REPO / "prep"))
    p.add_argument("--out-root", default=str(REPO / "outputs"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    amp = not args.no_amp
    device = torch.device(args.device)
    data, prep = Path(args.data), Path(args.prep)
    out = Path(args.out_root) / args.run_name
    out.mkdir(parents=True, exist_ok=args.resume)
    log = (out / "run.log").open("a")

    def say(*parts):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(p) for p in parts)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    seed_everything(args.seed)
    lr = args.lr if args.lr is not None else 0.02 * args.batch / 16
    lo, hi = args.test_size * (1 - args.jitter), args.test_size * (1 + args.jitter)
    train_sizes = list(range(int(round(lo / 32) * 32), int(hi) + 1, 32))
    max_size_train = int(math.ceil(args.max_size * (1 + args.jitter)))
    config = {**vars(args), "lr": lr, "train_short_sides": train_sizes, "max_size_train": max_size_train,
              "optimizer": "SGD momentum 0.9, wd 1e-4", "schedule": "linear warm-up, then per-iteration cosine to 0",
              "train_images": f"prep/yolo/train_step{args.step}_masked_fold{args.fold}_train.txt (masked)",
              "train_gt": f"prep/coco/train_step{args.step}_fold{args.fold}_train.json",
              "val_gt": f"prep/coco/train_step{args.step}_fold{args.fold}_val.json (original images)"}
    write_json(out / "config.json", config)

    gt_val = read_json(prep / "coco" / f"train_step{args.step}_fold{args.fold}_val.json")
    train_set = TrainSet(prep / "yolo" / f"train_step{args.step}_masked_fold{args.fold}_train.txt",
                         prep / "coco" / f"train_step{args.step}_fold{args.fold}_train.json")
    val_images = gt_val["images"]
    if args.limit:
        train_set.items = train_set.items[:args.limit]
        val_images = val_images[:args.limit]
        from .evaluate import subset_ground_truth
        gt_val = subset_ground_truth(gt_val, [i["id"] for i in val_images])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True, generator=generator, worker_init_fn=worker_init,
                        persistent_workers=args.workers > 0, pin_memory=True)

    model = build_model(args.arch, train_sizes, args.test_size, max_size_train, args.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    warmup = min(args.warmup_iters, total // 4)

    def factor(it):
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best, history = 0, -1.0, []
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, history = state["epoch"] + 1, state["best"], state["history"]
        random.setstate(state["py_random"])
        torch.set_rng_state(state["torch_random"])
        generator.set_state(state["loader_random"])
        say("resumed at epoch", start_epoch)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    write_json(out / "runtime.json", {
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(device), "commit": commit,
        "src_dirty": dirty, "versions": {p: version(p) for p in ("torch", "torchvision", "numpy", "faster-coco-eval")},
        "folds_sha256": hashlib.sha256((REPO / "folds.csv").read_bytes()).hexdigest()})
    say(f"{args.arch} | train {len(train_set)} imgs, {len(loader)} it/epoch x {args.epochs} epochs | "
        f"batch {args.batch} lr {lr} | train short sides {train_sizes[0]}..{train_sizes[-1]} | val {len(val_images)}")

    started = time.perf_counter()
    iteration = start_epoch * len(loader)
    for epoch in range(start_epoch, args.epochs):
        set_mode(model, True)
        sums, count, t0 = {}, 0, time.perf_counter()
        for images, targets in loader:
            images = [i.to(device, non_blocking=True) for i in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at iteration {iteration}: {losses}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            iteration += 1
            count += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            if count % 100 == 0:
                say(f"epoch {epoch + 1}/{args.epochs} it {count}/{len(loader)} loss {sums['loss'] / count:.4f} "
                    f"lr {optimizer.param_groups[0]['lr']:.5f} {count * args.batch / (time.perf_counter() - t0):.1f} img/s "
                    f"mem {torch.cuda.max_memory_reserved(device) / 2**30:.1f}GB")
        row = {"epoch": epoch + 1, **{k: v / count for k, v in sums.items()},
               "lr": optimizer.param_groups[0]["lr"], "epoch_min": (time.perf_counter() - t0) / 60}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            t1 = time.perf_counter()
            predictions = predict(model, val_images, data / "train", device, workers=args.workers, amp=amp)
            metrics = evaluate_predictions(gt_val, predictions)["metrics"]
            row.update({k: v for k, v in metrics.items()}, eval_min=(time.perf_counter() - t1) / 60)
            if metrics["mAP"] > best:
                best = metrics["mAP"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, out / "best.pt")
        history.append(row)
        say("epoch summary", json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}))
        with (out / "train_log.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for r in history for k in r))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best, "history": history, "py_random": random.getstate(),
                    "torch_random": torch.get_rng_state(), "loader_random": generator.get_state()}, out / "last.pt")
    training_hours = (time.perf_counter() - started) / 3600
    say(f"training done in {training_hours:.2f} h; best fold mAP {best:.4f}")

    # Final reported numbers come from the last epoch (fixed-length protocol), not the best checkpoint.
    predictions = predict(model, val_images, data / "train", device, workers=args.workers, amp=amp)
    write_json(out / "val_predictions.json", predictions)
    report = evaluate_predictions(gt_val, predictions)
    write_json(out / "metrics.json", report)
    say("final fold metrics", json.dumps(report["metrics"]))
    paths = [data / "train" / i["file_name"] for i in val_images]
    sample = [paths[round(i * (len(paths) - 1) / 31)] for i in range(32)] if len(paths) > 1 else paths
    speed = benchmark(model, sample, device, amp=amp)
    write_json(out / "speed.json", speed)
    say("speed", json.dumps(speed))
    if args.test:
        test_images = submit.read_test_images(data / "test_images.csv")
        if args.limit:  # debug only: partial submission, never upload it
            test_images = test_images[:args.limit]
        test_predictions = predict(model, test_images, data, device, workers=args.workers, amp=amp)
        write_json(out / "test_predictions.json", test_predictions)
        submit.write_submission(test_images, test_predictions, out / "submission.csv")
        check = submit.validate_submission(test_images, out / "submission.csv")
        write_json(out / "submission_check.json", check)
        say("submission", json.dumps(check))
    row = {"id": args.run_name, "owner": "Oleksandr", "family": args.arch, "fold": args.fold, **report["metrics"],
           "FPS": speed["fps"], "GPU_h_training": training_hours, "public_LB": "", "commit": commit}
    with (out / "experiment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    say("done")


if __name__ == "__main__":
    main()
