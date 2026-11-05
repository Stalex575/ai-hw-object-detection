"""Draw ground truth on sample frames: a visual check of labels and ignore regions.

Boxes are coloured by class; ignore regions are drawn as grey outlines with a light fill.
Picks one frame (the middle one) from each of N clips, spread over folds and resolutions.

Usage:
    python src/viz_gt.py --split-dir train --folds folds.csv --out images/gt_samples.jpg
    python src/viz_gt.py --split-dir train --compare-dir prep/yolo/masked/images/train \
        --out images/mask_check.jpg --n 2       # original vs masked, side by side
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import class_names, list_frames, read_folds, read_norm_boxes, to_pixels  # noqa: E402

# Colour-blind-friendly palette (Okabe-Ito + extras), one per class.
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FFFFFF"]


def draw(img_path: Path, lbl: Path, ign: Path, names: list[str]) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for r in read_norm_boxes(ign, with_class=False):
        d.rectangle(to_pixels(r, W, H), fill=(255, 255, 255, 60), outline=(160, 160, 160, 255), width=2)
    lw = max(1, W // 640)
    for line in (lbl.read_text().split("\n") if lbl.exists() else []):
        if line.strip():
            c, *box = line.split()
            d.rectangle(to_pixels([float(v) for v in box], W, H), outline=COLORS[int(c) % 8], width=lw)
    im = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")
    return im


def legend(names: list[str], width: int) -> Image.Image:
    lg = Image.new("RGB", (width, 28), "black")
    d = ImageDraw.Draw(lg)
    x = 8
    for i, n in enumerate(names + ["ignore"]):
        col = COLORS[i] if i < len(names) else "#A0A0A0"
        d.rectangle([x, 8, x + 12, 20], outline=col, width=2)
        d.text((x + 16, 8), n, fill="white")
        x += 24 + 7 * len(n)
    return lg


def grid(tiles: list[Image.Image], cols: int, tile_w: int) -> Image.Image:
    tiles = [t.resize((tile_w, round(t.height * tile_w / t.width))) for t in tiles]
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * tile_w, rows * th), "black")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * tile_w, (i // cols) * th))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--folds", help="spread picks over folds")
    ap.add_argument("--compare-dir", help="masked images root: show original | masked pairs")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tile-width", type=int, default=960)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    split = Path(args.split_dir)
    names = class_names(split)
    clips = list_frames(split)
    folds = read_folds(Path(args.folds)) if args.folds else {}

    # Candidate frame per clip: the middle frame; with --compare-dir, prefer frames with ignore regions.
    cands = []
    for clip, frames in clips.items():
        pool = frames
        if args.compare_dir:
            pool = [p for p in frames if (split / "ignore" / clip / f"{p.stem}.txt").exists()
                    and (Path(args.compare_dir) / clip / p.name).exists()]
            if not pool:
                continue
        p = pool[len(pool) // 2]
        with Image.open(p) as im:
            size = im.size
        cands.append((clip, p, size))
    # Spread: round-robin over (fold, resolution) groups, deterministic.
    groups: dict = {}
    for c in cands:
        groups.setdefault((folds.get(c[0], 0), c[2]), []).append(c)
    # Order groups so consecutive picks alternate folds: (rank of resolution within fold, fold).
    rank = {}
    for f in sorted({k[0] for k in groups}):
        for r, k in enumerate(sorted(k for k in groups if k[0] == f)):
            rank[k] = (r, f)
    picks, keys = [], sorted(groups, key=rank.get)
    while len(picks) < args.n and any(groups.values()):
        for k in keys:
            if groups[k] and len(picks) < args.n:
                picks.append(groups[k].pop(len(groups[k]) // 2))

    tiles = []
    for clip, p, _ in picks:
        lbl = split / "labels" / clip / f"{p.stem}.txt"
        ign = split / "ignore" / clip / f"{p.stem}.txt"
        tiles.append(draw(p, lbl, ign, names))
        if args.compare_dir:
            tiles.append(draw(Path(args.compare_dir) / clip / p.name, lbl, ign, names))
        print(f"{clip}/{p.name} fold={folds.get(clip, '-')} size={_[0]}x{_[1]}", file=sys.stderr)

    cols = 2 if args.compare_dir or len(tiles) > 3 else len(tiles)
    g = grid(tiles, cols, args.tile_width)
    out = Image.new("RGB", (g.width, g.height + 28), "black")
    out.paste(g, (0, 0))
    out.paste(legend(names, g.width), (0, g.height))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, quality=90)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
