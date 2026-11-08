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

    def __init__(self, image_list, coco_json, hflip=0.5, lowres_p=0.0, lowres_widths=(640, 1000)):
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
        self.lowres_p, self.lowres_widths = lowres_p, lowres_widths

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
        if self.lowres_p and random.random() < self.lowres_p:
            # Low-resolution augmentation: the test set has 680x382 / 960x540 clips that training lacks. Shrink the
            # frame to a test-like width and blow it back up to its own size, so boxes stay valid and only detail is lost.
            target_w = random.uniform(*self.lowres_widths)
            if width > 1.1 * target_w:
                small = (max(1, round(height * target_w / width)), round(target_w))
                x = torch.nn.functional.interpolate(image[None], size=small, mode="bilinear", antialias=True)
                image = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear")[0].clamp_(0, 1)
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


def explicit_anchor_generator(wh_per_level):
    """AnchorGenerator whose per-level cell anchors are explicit (w, h) pairs, e.g. from k-means."""
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    generator = AnchorGenerator(tuple((1,) * len(level) for level in wh_per_level),
                                ((1.0,),) * len(wh_per_level))  # sizes/ratios here only fix the anchor counts
    generator.cell_anchors = [torch.tensor([[-w / 2, -h / 2, w / 2, h / 2] for w, h in level]).round()
                              for level in wh_per_level]
    return generator


def fastrcnn_loss_giou(weight):
    """RoI-head loss with GIoU on decoded boxes instead of Smooth L1 on encoded deltas.

    Prediction and target deltas are both decoded against a unit reference box. IoU and GIoU are invariant
    to per-axis scaling and translation, so this equals GIoU on the boxes decoded against the real proposals
    (which torchvision's fastrcnn_loss does not receive). Weight 10 follows mmdetection's Faster R-CNN GIoU config.
    """
    import torch.nn.functional as F
    from torchvision.models.detection._utils import BoxCoder
    from torchvision.ops import generalized_box_iou_loss

    coder = BoxCoder((10.0, 10.0, 5.0, 5.0))  # torchvision RoIHeads default bbox_reg_weights

    def loss_fn(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        positive = torch.where(labels > 0)[0]
        box_regression = box_regression.reshape(class_logits.shape[0], -1, 4)
        pred = box_regression[positive, labels[positive]].float()
        target = regression_targets[positive].float()
        unit = pred.new_tensor([[0.0, 0.0, 1.0, 1.0]]).expand(len(positive), 4)
        box_loss = generalized_box_iou_loss(coder.decode_single(pred, unit), coder.decode_single(target, unit),
                                            reduction="sum")
        return classification_loss, weight * box_loss / labels.numel()

    return loss_fn


def configure(model, arch, args):
    """Apply the single-knob experiment flags. Returns a dict describing what changed (for config.json)."""
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    changed = {}
    generator = None
    if args.anchors_json:
        spec = read_json(args.anchors_json)
        generator = explicit_anchor_generator(spec["wh"])
        changed["anchors"] = spec["wh"]
    elif args.anchor_sizes:
        sizes = [int(x) for x in args.anchor_sizes.split(",")]
        if arch == "retinanet":  # keep torchvision's 3 octave scales per level
            per_level = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in sizes)
        else:
            per_level = tuple((x,) for x in sizes)
        generator = AnchorGenerator(per_level, ((0.5, 1.0, 2.0),) * len(per_level))
        changed["anchor_sizes"] = per_level
    if generator is not None:
        old = model.anchor_generator if arch == "retinanet" else model.rpn.anchor_generator
        if generator.num_anchors_per_location() != old.num_anchors_per_location():
            raise ValueError(f"anchors per location {generator.num_anchors_per_location()} != "
                             f"{old.num_anchors_per_location()} expected by the pretrained head")
        if arch == "retinanet":
            model.anchor_generator = generator
        else:
            model.rpn.anchor_generator = generator
    if args.fg_iou is not None or args.bg_iou is not None:
        if arch == "retinanet":
            fg, bg = args.fg_iou or 0.5, args.bg_iou or 0.4
            model.proposal_matcher = det_utils.Matcher(fg, bg, allow_low_quality_matches=True)
        else:
            fg, bg = args.fg_iou or 0.5, args.bg_iou or 0.5
            model.roi_heads.proposal_matcher = det_utils.Matcher(fg, bg, allow_low_quality_matches=False)
        changed.update(fg_iou=fg, bg_iou=bg)
    if args.focal_gamma is not None:
        if arch != "retinanet":
            raise ValueError("--focal-gamma is RetinaNet only")
        import functools
        from torchvision.models.detection import retinanet as R
        from torchvision.ops import sigmoid_focal_loss
        R.sigmoid_focal_loss = functools.partial(sigmoid_focal_loss, alpha=0.25, gamma=args.focal_gamma)
        changed["focal_gamma"] = args.focal_gamma
    if args.box_positive_fraction is not None:
        if arch != "fasterrcnn":
            raise ValueError("--box-positive-fraction is Faster R-CNN only")
        model.roi_heads.fg_bg_sampler.positive_fraction = args.box_positive_fraction
        changed["box_positive_fraction"] = args.box_positive_fraction
    if args.box_loss == "giou":
        if arch != "fasterrcnn":
            raise ValueError("--box-loss giou is Faster R-CNN only (RetinaNet v2 already uses GIoU)")
        from torchvision.models.detection import roi_heads
        roi_heads.fastrcnn_loss = fastrcnn_loss_giou(args.giou_weight)
        changed.update(box_loss="giou", giou_weight=args.giou_weight)
    if args.nms_thresh is not None:
        if arch == "retinanet":
            model.nms_thresh = args.nms_thresh
        else:
            model.roi_heads.nms_thresh = args.nms_thresh
        changed["nms_thresh"] = args.nms_thresh
    if args.rpn_post_nms_test is not None:
        model.rpn._post_nms_top_n["testing"] = args.rpn_post_nms_test
        changed["rpn_post_nms_test"] = args.rpn_post_nms_test
    return changed


def set_mode(model, train):
    """Switch between scale-jitter training sizes and the fixed test size."""
    sizes, max_size = model.size_cfg["train" if train else "test"]
    model.transform.min_size, model.transform.max_size = sizes, max_size
    model.train(train)


@torch.no_grad()
def predict(model, images, root, device, batch=4, workers=4, amp=True, hflip=False):
    """hflip: test-time augmentation pass on mirrored frames; boxes are mirrored back to the original frame."""
    set_mode(model, False)
    loader = DataLoader(ImageSet(images, root), batch_size=batch, num_workers=workers, collate_fn=collate)
    predictions, seen = [], 0
    for batch_images, infos in loader:
        if hflip:
            batch_images = [i.flip(-1) for i in batch_images]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outputs = model([i.to(device, non_blocking=True) for i in batch_images])
        for output, info, image in zip(outputs, infos, batch_images):
            boxes = output["boxes"].float().cpu()
            if hflip:
                width = image.shape[-1]
                boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1)
            boxes = boxes.tolist()
            for (x1, y1, x2, y2), score, label in zip(boxes, output["scores"].float().cpu().tolist(),
                                                     output["labels"].cpu().tolist()):
                # RetinaNet keeps an unused logit for label 0 (no positives ever); drop it, it is not a class.
                if x2 > x1 and y2 > y1 and label >= 1:
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
    p.add_argument("--fold", type=int, default=0,
                   help="-1 = final model: train on ALL training clips, score on the untouched official val/ split")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--lowres-p", type=float, default=0.0,
                   help="probability of the low-resolution augmentation (shrink to 640-1000 px wide, enlarge back)")
    p.add_argument("--no-mask", action="store_true",
                   help="D1 ablation: train on the original frames instead of the ignore-region-masked ones")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="default: 0.02 * batch / 16 (SGD)")
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--test-size", type=int, default=720, help="short side at test time (16:9 -> 1280 long side)")
    p.add_argument("--max-size", type=int, default=1280, help="long-side cap at test time")
    p.add_argument("--jitter", type=float, default=0.25, help="train short side in [1-j, 1+j] * test-size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=None,
                   help="loader workers for val/test inference (default: --workers). Fewer = lower RAM peak at eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=2, help="shared-evaluator fold-0 mAP every N epochs")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--test", action="store_true", help="also predict the test set and write submission.csv")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N train/val images")
    g = p.add_argument_group("single-knob experiment flags (default = torchvision behaviour)")
    g.add_argument("--anchor-sizes", help="comma list of base anchor sizes per FPN level, e.g. 16,32,64,128,256")
    g.add_argument("--anchors-json", help='explicit per-level anchors {"wh": [[[w, h], ...], ...]} (k-means)')
    g.add_argument("--fg-iou", type=float, help="RetinaNet anchor / Faster R-CNN RoI foreground IoU")
    g.add_argument("--bg-iou", type=float, help="RetinaNet anchor / Faster R-CNN RoI background IoU")
    g.add_argument("--focal-gamma", type=float, help="RetinaNet focal loss gamma (default 2.0)")
    g.add_argument("--box-positive-fraction", type=float, help="Faster R-CNN RoI sampler positive fraction (0.25)")
    g.add_argument("--box-loss", choices=["smoothl1", "giou"], default="smoothl1", help="Faster R-CNN RoI box loss")
    g.add_argument("--giou-weight", type=float, default=10.0)
    g.add_argument("--nms-thresh", type=float, help="final per-class NMS IoU (Faster R-CNN 0.5, RetinaNet 0.5)")
    g.add_argument("--rpn-post-nms-test", type=int, help="Faster R-CNN RPN proposals kept at test time (1000)")
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
    if args.fold < 0:  # final model: every training clip; official val/ (step 5) as the only holdout
        train_list = prep / "yolo" / (f"train_step{args.step}.txt" if args.no_mask else f"train_step{args.step}_masked.txt")
        train_gt, val_gt, val_root = f"train_step{args.step}.json", "val_step5.json", data / "val"
    else:
        train_list = prep / "yolo" / (f"train_step{args.step}_fold{args.fold}_train.txt" if args.no_mask
                                      else f"train_step{args.step}_masked_fold{args.fold}_train.txt")
        train_gt = f"train_step{args.step}_fold{args.fold}_train.json"
        val_gt, val_root = f"train_step{args.step}_fold{args.fold}_val.json", data / "train"
    lo, hi = args.test_size * (1 - args.jitter), args.test_size * (1 + args.jitter)
    train_sizes = list(range(int(round(lo / 32) * 32), int(hi) + 1, 32))
    max_size_train = int(math.ceil(args.max_size * (1 + args.jitter)))
    config = {**vars(args), "lr": lr, "train_short_sides": train_sizes, "max_size_train": max_size_train,
              "optimizer": "SGD momentum 0.9, wd 1e-4", "schedule": "linear warm-up, then per-iteration cosine to 0",
              "train_images": train_list.name + (" (ORIGINAL, unmasked)" if args.no_mask else " (masked)"),
              "train_gt": f"prep/coco/{train_gt}", "val_gt": f"prep/coco/{val_gt} (original images)"}
    write_json(out / "config.json", config)

    gt_val = read_json(prep / "coco" / val_gt)
    train_set = TrainSet(train_list, prep / "coco" / train_gt, lowres_p=args.lowres_p)
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

    model = build_model(args.arch, train_sizes, args.test_size, max_size_train, args.max_size)
    config["changes"] = configure(model, args.arch, args)
    write_json(out / "config.json", config)
    model = model.to(device)
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
    if config["changes"]:
        say("changes vs baseline:", json.dumps(config["changes"]))
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
            predictions = predict(model, val_images, val_root, device, workers=args.eval_workers if args.eval_workers is not None else args.workers, amp=amp)
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
    predictions = predict(model, val_images, val_root, device, workers=args.eval_workers if args.eval_workers is not None else args.workers, amp=amp)
    write_json(out / "val_predictions.json", predictions)
    report = evaluate_predictions(gt_val, predictions)
    write_json(out / "metrics.json", report)
    say("final fold metrics", json.dumps(report["metrics"]))
    paths = [val_root / i["file_name"] for i in val_images]
    sample = [paths[round(i * (len(paths) - 1) / 31)] for i in range(32)] if len(paths) > 1 else paths
    speed = benchmark(model, sample, device, amp=amp)
    write_json(out / "speed.json", speed)
    say("speed", json.dumps(speed))
    if args.test:
        test_images = submit.read_test_images(data / "test_images.csv")
        if args.limit:  # debug only: partial submission, never upload it
            test_images = test_images[:args.limit]
        test_predictions = predict(model, test_images, data, device, workers=args.eval_workers if args.eval_workers is not None else args.workers, amp=amp)
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
