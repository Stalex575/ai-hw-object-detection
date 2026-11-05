"""Clip-level folds for the train split: GroupKFold(n_splits=3) grouped by clip.

Frames of one clip are near-duplicates, so a frame-level split would leak. GroupKFold
puts every clip in exactly one fold and balances the number of frames per fold. It has
no randomness, so folds.csv is reproducible from the data alone.

Usage:
    python src/folds.py --split-dir DATA/train --out folds.csv [--n-splits 3]

Output columns: clip, fold, n_frames, n_boxes, then one box-count column per class.
Fold 0 is the fixed tuning split (its clips are the validation part).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import class_names, list_frames  # noqa: E402


def clip_stats(split_dir: Path, n_classes: int) -> list[dict]:
    rows = []
    for clip, frames in list_frames(split_dir).items():
        counts = [0] * n_classes
        for p in frames:
            lbl = split_dir / "labels" / clip / f"{p.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text().split("\n"):
                    if line.strip():
                        counts[int(line.split()[0])] += 1
        rows.append({"clip": clip, "n_frames": len(frames), "counts": counts})
    return rows


def assign_folds(rows: list[dict], n_splits: int) -> dict[str, int]:
    from sklearn.model_selection import GroupKFold
    # One sample per frame, grouped by clip, so folds are balanced by frame count.
    groups = [r["clip"] for r in rows for _ in range(r["n_frames"])]
    fold_of = {}
    for k, (_, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(groups, groups=groups)):
        for i in val_idx:
            fold_of[groups[i]] = k
    return fold_of


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, help="e.g. DATA/train")
    ap.add_argument("--out", default="folds.csv")
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args(argv)

    split_dir = Path(args.split_dir)
    names = class_names(split_dir)
    rows = clip_stats(split_dir, len(names))
    fold_of = assign_folds(rows, args.n_splits)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "fold", "n_frames", "n_boxes", *names])
        for r in sorted(rows, key=lambda r: (fold_of[r["clip"]], r["clip"])):
            w.writerow([r["clip"], fold_of[r["clip"]], r["n_frames"], sum(r["counts"]), *r["counts"]])
    print(f"wrote {args.out}", file=sys.stderr)

    # Balance check: frames, boxes and class shares per fold.
    hdr = f"{'fold':>4} {'clips':>5} {'frames':>7} {'boxes':>8}  " + " ".join(f"{n[:7]:>7}" for n in names)
    print(hdr)
    for k in range(args.n_splits):
        rs = [r for r in rows if fold_of[r["clip"]] == k]
        counts = [sum(r["counts"][c] for r in rs) for c in range(len(names))]
        tot = max(1, sum(counts))
        print(f"{k:>4} {len(rs):>5} {sum(r['n_frames'] for r in rs):>7} {sum(counts):>8}  "
              + " ".join(f"{100 * c / tot:>6.1f}%" for c in counts))


if __name__ == "__main__":
    main()
