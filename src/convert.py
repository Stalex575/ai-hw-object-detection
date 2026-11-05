"""Prepare the competition data (VisDrone2019-VID, 8 classes) for training and evaluation.

The competition already ships YOLO labels, COCO JSON and ignore regions:

    <split>/images/<clip>/<frame>.jpg
    <split>/labels/<clip>/<frame>.txt      class cx cy w h (normalised, class 0..7)
    <split>/ignore/<clip>/<frame>.txt      cx cy w h (normalised), only frames that have them
    <split>/annotations_coco.json          category_id 1..8, ignore regions as iscrowd=1

This script adds what the shared protocol needs on top of that:

  coco  Evaluator-ready GT JSON. COCO eval matches crowd regions *per category*, so every
        ignore region must exist once per class; then a detection of ANY class inside it
        is ignored, as in VisDrone scoring. The supplied JSON already stores one copy per
        class (checked on val); identical copies are merged and re-expanded, so this is a
        no-op there and only fills in missing copies. Optionally writes frame-step and
        per-fold subsets.

  yolo  Ultralytics image lists (per fold, with frame step) and data yamls. With --mask
        (default) frames that have ignore regions are rewritten with those regions filled
        with the image mean pixel (pixels of real boxes overlapping them are kept); all
        other frames and all labels are symlinked, so the copy stays small.

Usage:
    python src/convert.py coco --split-dir DATA/train --out prep/coco/train.json --folds folds.csv
    python src/convert.py coco --split-dir DATA/val   --out prep/coco/val.json
    python src/convert.py coco --split-dir DATA/val   --out prep/coco/val_step5.json --frame-step 5
    python src/convert.py yolo --split-dir DATA/train --out prep/yolo --folds folds.csv --frame-step 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- dataset index

def list_frames(split_dir: Path) -> dict[str, list[Path]]:
    """clip -> image paths sorted by frame number."""
    img_root = split_dir / "images"
    if not img_root.is_dir():
        sys.exit(f"error: {img_root} not found")
    clips = {}
    for clip_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        frames = [p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        clips[clip_dir.name] = sorted(frames, key=lambda p: (len(p.stem), p.stem))
    return clips


def step_filter(paths: list[Path], frame_step: int) -> list[Path]:
    """Every N-th frame of a clip, starting from the first."""
    return paths[::frame_step]


def read_folds(path: Path) -> dict[str, int]:
    with open(path) as f:
        return {r["clip"]: int(r["fold"]) for r in csv.DictReader(f)}


def class_names(split_dir: Path) -> list[str]:
    """Class names from data.yaml (list or {id: name}); falls back to the documented list."""
    y = split_dir / "data.yaml"
    if y.exists():
        try:
            import yaml
            names = yaml.safe_load(y.read_text()).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            if names:
                return list(names)
        except ImportError:
            pass
    return DEFAULT_CLASSES


# --------------------------------------------------------------------------- COCO

def image_clip(img: dict) -> str:
    """Clip name of a COCO image entry: the `video` field, else the parent dir of file_name."""
    return str(img.get("video") or Path(img["file_name"]).parent.name)


def image_frame_key(img: dict) -> tuple:
    fid = img.get("frame_id")
    if fid is not None:
        return (0, int(fid), "")
    stem = Path(img["file_name"]).stem
    return (1, len(stem), stem)


def expand_crowd(coco: dict) -> tuple[dict, int]:
    """Give every iscrowd=1 region one copy per category. Returns (new coco, n_regions)."""
    cat_ids = sorted(c["id"] for c in coco["categories"])
    next_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    out, seen, n_regions = [], set(), 0
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            out.append(a)
            continue
        key = (a["image_id"], tuple(round(v, 2) for v in a["bbox"]))
        if key in seen:  # already expanded (e.g. the source stored one copy per class)
            continue
        seen.add(key)
        n_regions += 1
        for cid in cat_ids:
            b = dict(a, category_id=cid, id=next_id)
            b.setdefault("area", a["bbox"][2] * a["bbox"][3])
            out.append(b)
            next_id += 1
    return dict(coco, annotations=out), n_regions


def subset_coco(coco: dict, image_ids: set) -> dict:
    return dict(coco,
                images=[i for i in coco["images"] if i["id"] in image_ids],
                annotations=[a for a in coco["annotations"] if a["image_id"] in image_ids])


def stepped_image_ids(coco: dict, frame_step: int) -> set:
    by_clip = defaultdict(list)
    for img in coco["images"]:
        by_clip[image_clip(img)].append(img)
    keep = set()
    for imgs in by_clip.values():
        imgs.sort(key=image_frame_key)
        keep.update(i["id"] for i in imgs[::frame_step])
    return keep


def write_json(coco: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)
    n_real = sum(1 for a in coco["annotations"] if not a.get("iscrowd"))
    print(f"wrote {path}: {len(coco['images'])} images, {n_real} boxes, "
          f"{len(coco['annotations']) - n_real} crowd anns", file=sys.stderr)


def cmd_coco(args) -> None:
    split_dir, out = Path(args.split_dir), Path(args.out)
    with open(split_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    crowd_cats = defaultdict(int)
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            crowd_cats[a.get("category_id")] += 1
    print(f"source: {len(coco['images'])} images, crowd anns by category_id: {dict(crowd_cats)}",
          file=sys.stderr)

    coco, n_regions = expand_crowd(coco)
    print(f"expanded {n_regions} ignore regions x {len(coco['categories'])} classes", file=sys.stderr)
    coco = subset_coco(coco, stepped_image_ids(coco, args.frame_step))
    coco.setdefault("info", {})["prepared_by"] = (
        f"src/convert.py coco frame_step={args.frame_step}, crowd regions expanded to all classes")
    write_json(coco, out)

    if args.folds:
        folds = read_folds(Path(args.folds))
        clips = {image_clip(i) for i in coco["images"]}
        if clips - set(folds):
            sys.exit(f"error: clips missing from {args.folds}: {sorted(clips - set(folds))[:5]}")
        for k in sorted(set(folds.values())):
            for role in ("train", "val"):
                ids = {i["id"] for i in coco["images"]
                       if (folds[image_clip(i)] == k) == (role == "val")}
                write_json(subset_coco(coco, ids), out.with_name(f"{out.stem}_fold{k}_{role}.json"))


# --------------------------------------------------------------------------- YOLO

def read_norm_boxes(path: Path, with_class: bool) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().split("\n"):
        v = line.split()
        if v:
            rows.append([float(x) for x in v[1:5]] if with_class else [float(x) for x in v[:4]])
    return rows


def to_pixels(cxcywh: list[float], W: int, H: int) -> tuple[int, int, int, int]:
    """Normalised cx cy w h -> clipped integer x1 y1 x2 y2 (covering the box)."""
    import math
    cx, cy, w, h = cxcywh
    x1, y1 = max(0, math.floor((cx - w / 2) * W)), max(0, math.floor((cy - h / 2) * H))
    x2, y2 = min(W, math.ceil((cx + w / 2) * W)), min(H, math.ceil((cy + h / 2) * H))
    return x1, y1, x2, y2


def write_masked(img_path: Path, ignore: list, boxes: list, dst: Path) -> None:
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    mask = np.zeros((H, W), bool)
    for r in ignore:
        x1, y1, x2, y2 = to_pixels(r, W, H)
        mask[y1:y2, x1:x2] = True
    for b in boxes:  # never erase a labelled object that overlaps an ignore region
        x1, y1, x2, y2 = to_pixels(b, W, H)
        mask[y1:y2, x1:x2] = False
    img[mask] = img.reshape(-1, 3).mean(0).round().astype(np.uint8)
    Image.fromarray(img).save(dst, quality=95)


def symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.absolute(), dst)


def yaml_text(root: Path, train: str, val: str, names: list[str]) -> str:
    lines = [f"path: {root.resolve()}", f"train: {train}", f"val: {val}", "names:"]
    return "\n".join(lines + [f"  {i}: {n}" for i, n in enumerate(names)]) + "\n"


def cmd_yolo(args) -> None:
    split_dir, out = Path(args.split_dir), Path(args.out)
    split = split_dir.name
    clips = list_frames(split_dir)
    names = class_names(split_dir)
    folds = read_folds(Path(args.folds)) if args.folds else None
    if folds and set(clips) - set(folds):
        sys.exit(f"error: clips missing from {args.folds}: {sorted(set(clips) - set(folds))[:5]}")

    # Ultralytics finds labels by replacing the last /images/ with /labels/ in the listed
    # (unresolved) path, so the masked copy is masked/{images,labels}/<split>/<clip>/...
    img_root = out / "masked" / "images" / split
    lbl_root = out / "masked" / "labels" / split
    listed: dict[str, list[str]] = {}
    n_frames = n_masked = 0
    for clip, paths in clips.items():
        listed[clip] = []
        for p in step_filter(paths, args.frame_step):
            n_frames += 1
            if not args.mask:  # list the original files; labels sit next to them
                listed[clip].append(str(p.absolute()))
                continue
            (img_root / clip).mkdir(parents=True, exist_ok=True)
            (lbl_root / clip).mkdir(parents=True, exist_ok=True)
            lbl = split_dir / "labels" / clip / f"{p.stem}.txt"
            ign = split_dir / "ignore" / clip / f"{p.stem}.txt"
            dst = img_root / clip / p.name
            ignore = read_norm_boxes(ign, with_class=False)
            if ignore:
                if dst.is_symlink():
                    dst.unlink()
                if args.overwrite or not dst.exists():
                    write_masked(p, ignore, read_norm_boxes(lbl, with_class=True), dst)
                n_masked += 1
            else:
                symlink(p, dst)
            if lbl.exists():
                symlink(lbl, lbl_root / clip / lbl.name)
            listed[clip].append(str(dst.absolute()))
            if n_frames % 2000 == 0:
                print(f"  {n_frames} frames", file=sys.stderr)
    tag = f"{split}_step{args.frame_step}{'_masked' if args.mask else ''}"
    print(f"{tag}: {n_frames} frames, {n_masked} masked", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    all_list = out / f"{tag}.txt"
    all_list.write_text("\n".join(x for c in listed for x in listed[c]) + "\n")
    (out / f"{tag}.yaml").write_text(yaml_text(out, all_list.name, all_list.name, names))
    if folds:
        for k in sorted(set(folds.values())):
            for role in ("train", "val"):
                sel = [x for c in listed if (folds[c] == k) == (role == "val") for x in listed[c]]
                (out / f"{tag}_fold{k}_{role}.txt").write_text("\n".join(sel) + "\n")
            (out / f"{tag}_fold{k}.yaml").write_text(
                yaml_text(out, f"{tag}_fold{k}_train.txt", f"{tag}_fold{k}_val.txt", names))
        print(f"wrote {tag}_fold{{k}}.yaml for folds {sorted(set(folds.values()))}", file=sys.stderr)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split-dir", required=True, help="e.g. DATA/train (has images/, labels/, ...)")
    common.add_argument("--out", required=True)
    common.add_argument("--frame-step", type=int, default=1, help="keep every N-th frame of each clip")
    common.add_argument("--folds", help="folds.csv from src/folds.py; also write per-fold subsets")

    p = sub.add_parser("coco", parents=[common], help="evaluator-ready COCO GT (+ fold subsets)")
    p.set_defaults(fn=cmd_coco)

    p = sub.add_parser("yolo", parents=[common], help="Ultralytics lists/yamls (+ masked images)")
    p.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True,
                   help="fill ignore regions with the mean pixel (default: on)")
    p.add_argument("--overwrite", action="store_true", help="re-render masked images that exist")
    p.set_defaults(fn=cmd_yolo)

    args = ap.parse_args(argv)
    if args.frame_step < 1:
        ap.error("--frame-step must be >= 1")
    args.fn(args)


if __name__ == "__main__":
    main()
