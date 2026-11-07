"""A4: gather checkpoints + prediction JSONs into outputs/share/<run>/ for the Google Drive upload (plan section 6).

Layout matches the laptop so Oleh can drop it under outputs/:
  Ultralytics runs: train/weights/last.pt, last/{val,test}_predictions.json (last.pt re-evaluation, protocol rule 4)
  torchvision runs: last.pt (model weights only, no optimizer state), {val,test}_predictions.json, config.json
plus metrics.json / speed.json / experiment.csv, and predict_*/ folders (e.g. all-frame fold-0 predictions).
Files are hard-linked (no extra disk); the torchvision last.pt is rewritten without optimizer state.

    python tools/queue/make_share.py run_a run_b ...
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT, SHARE = REPO / "outputs", REPO / "outputs/share"
SMALL = ["metrics.json", "speed.json", "experiment.csv", "config.json", "train_log.csv", "submission_check.json",
         "val_predictions.json", "test_predictions.json", "training.json", "runtime.json"]


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(runs):
    import torch
    for run in runs:
        src, dst = OUT / run, SHARE / run
        for name in SMALL:
            if (src / name).exists():
                link(src / name, dst / name)
        for sub in sorted(p for p in src.iterdir() if p.is_dir() and (p.name == "last" or p.name.startswith("predict_"))):
            for f in sub.iterdir():
                if f.suffix in (".json", ".csv") and f.name != "submission.csv":
                    link(f, dst / sub.name / f.name)
        if (src / "train/weights/last.pt").exists():
            link(src / "train/weights/last.pt", dst / "train/weights/last.pt")
        elif (src / "last.pt").exists():
            state = torch.load(src / "last.pt", map_location="cpu", weights_only=False)
            torch.save({"model": state["model"], "epoch": state["epoch"]}, dst / "last.pt")
            del state
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2**20
        print(f"{run}: {size:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
