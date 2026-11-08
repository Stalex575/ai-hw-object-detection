"""Re-run inference with a trained torchvision checkpoint (outputs/<run>/last.pt), optionally with changed
inference-only settings, on the fold-0 val frames, any other COCO GT image set, or the test set.

The model is rebuilt from the run's config.json, so training-time structure changes (e.g. anchors) are kept.

    # all fold-0 val frames (not only every 5th), for the temporal post-processing
    python -m src.tv_predict frcnn_base_fold0_1280_e12_b2_s42_v1 --gt prep/coco/train_fold0_val.json --tag allframes
    # inference variant on the usual fold-0 frames
    python -m src.tv_predict frcnn_base_fold0_1280_e12_b2_s42_v1 --nms-thresh 0.6 --tag nms0.6
    # test predictions + submission for the same variant
    python -m src.tv_predict frcnn_base_fold0_1280_e12_b2_s42_v1 --nms-thresh 0.6 --tag nms0.6 --test

Outputs go to outputs/<run>/predict_<tag>/: val_predictions.json + metrics.json (when a GT is given), or
test_predictions.json + submission.csv + submission_check.json (with --test).
"""

import argparse
import json
from pathlib import Path

import torch

from . import submit
from .evaluate import evaluate_predictions, read_json, subset_ground_truth, write_json
from .tv_train import REPO, build_model, configure, parse_args as train_args, predict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run name under outputs/ or a path to the run directory")
    p.add_argument("--weights", default="last.pt")
    p.add_argument("--tag", required=True, help="output folder suffix: <run>/predict_<tag>/")
    p.add_argument("--gt", default=str(REPO / "prep/coco/train_step5_fold0_val.json"))
    p.add_argument("--test", action="store_true", help="predict test_images.csv instead of a GT image set")
    p.add_argument("--image-root", default="train", help="split folder the GT file_names are relative to (train or val)")
    p.add_argument("--nms-thresh", type=float)
    p.add_argument("--rpn-post-nms-test", type=int)
    p.add_argument("--test-size", type=int, help="short side at test time (training default 720)")
    p.add_argument("--max-size", type=int, help="long-side cap at test time (training default 1280)")
    p.add_argument("--hflip", action="store_true", help="test-time augmentation: predict on mirrored frames")
    p.add_argument("--min-score", type=float, default=0.0, help="drop detections below this score (smaller dumps)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    run = Path(args.run) if Path(args.run).is_dir() else REPO / "outputs" / args.run
    saved = read_json(run / "config.json")
    # Rebuild the training-time argument namespace, then apply the inference-only overrides.
    keep = {k: v for k, v in saved.items() if k in vars(train_args(["--run-name", "x", "--epochs", "1"]))}
    targs = argparse.Namespace(**{**vars(train_args(["--run-name", "x", "--epochs", "1"])), **keep})
    for key in ("nms_thresh", "rpn_post_nms_test"):
        if getattr(args, key) is not None:
            setattr(targs, key, getattr(args, key))
    test_size = args.test_size or targs.test_size
    max_size = args.max_size or targs.max_size
    model = build_model(targs.arch, saved["train_short_sides"], test_size, saved["max_size_train"], max_size)
    changes = configure(model, targs.arch, targs)
    state = torch.load(run / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    out = run / f"predict_{args.tag}"
    out.mkdir(exist_ok=True)
    info = {"run": run.name, "weights": args.weights, "arch": targs.arch, "test_size": test_size,
            "max_size": max_size, "changes": changes, "argv": vars(args)}
    if args.test:
        images = submit.read_test_images(REPO / "test_images.csv")
        if args.limit:
            images = images[:args.limit]
        predictions = predict(model, images, REPO, device, batch=args.batch, workers=args.workers, hflip=args.hflip)
        predictions = [d for d in predictions if d["score"] >= args.min_score]
        write_json(out / "test_predictions.json", predictions)
        submit.write_submission(images, predictions, out / "submission.csv")
        check = submit.validate_submission(images, out / "submission.csv")
        write_json(out / "submission_check.json", check)
        info["submission"] = check
    else:
        gt = read_json(args.gt)
        if args.limit:
            gt = subset_ground_truth(gt, [i["id"] for i in gt["images"][:args.limit]])
        predictions = predict(model, gt["images"], REPO / args.image_root, device, batch=args.batch, workers=args.workers, hflip=args.hflip)
        predictions = [d for d in predictions if d["score"] >= args.min_score]
        write_json(out / "val_predictions.json", predictions)
        report = evaluate_predictions(gt, predictions)
        report["gt"] = str(Path(args.gt).resolve().relative_to(REPO)) if Path(args.gt).resolve().is_relative_to(REPO) else args.gt
        write_json(out / "metrics.json", report)
        info["metrics"] = report["metrics"]
    write_json(out / "predict_info.json", info)
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
