"""WBF + temporal rescoring of test predictions, one clip at a time (low RAM; same result as the whole-set run,
since both steps only look inside one image / one clip)."""
import csv, gc, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from src import submit
from src.ensemble import fuse, compact
from src.temporal import rescore
from src.evaluate import read_json, write_json, normalize_predictions
F = "outputs/frcnn_final_all_step2_lowres_e6_b2_s42_v1"; Y = "outputs/yolo26s_final_all_step2_e12_b3_s42_v1"
O = "outputs/oleh_yolo26s_augs_epoch5"; P = "outputs/oleh_yolo26s_final_all_e5"
base = [f"{F}/test_predictions.json", f"{F}/predict_test_hflip/test_predictions.json", f"{Y}/test_best.json", f"{Y}/test_best_hflip.json",
        f"{F}/predict_e2_test/test_predictions.json", f"{F}/predict_e2_test_hflip/test_predictions.json", f"{Y}/test_epoch2.json",
        f"{Y}/test_epoch2_hflip.json", f"{O}/test_best.json", f"{O}/test_best_hflip.json"]
mode = sys.argv[1]
if mode == "check10":
    files, weights, out, name = base, [1] * 8 + [2, 2], REPO / "outputs/ens10_oleh_w2_iou0.6", "perclip_check.csv"
else:
    W = int(mode)
    files = base + [f"{P}/test_best.json", f"{P}/test_best_hflip.json"]
    weights, out = [1] * 8 + [2, 2, W, W], REPO / f"outputs/ens12_oleh2_w{W}_iou0.6"
    name = f"submission_ens12_oleh2_w{W}_boxboost.csv"
out.mkdir(exist_ok=True)
images = submit.read_test_images(REPO / "test_images.csv")
parts = []
for f in files:
    parts.append(compact(read_json(REPO / f), images)); gc.collect()
clips = {}
for i, image in enumerate(images):
    clips.setdefault(image["video"], []).append(i)
order = [i for idx in clips.values() for i in idx]
assert order == list(range(len(images))), "manifest is not grouped by clip"
with (out / name).open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["image_id", "PredictionString"])
    for clip, idx in clips.items():
        sub_images = [images[i] for i in idx]
        remap = np.full(len(images), -1, np.int64); remap[idx] = np.arange(len(idx))
        sub = []
        for image_index, category, score, boxes in parts:
            sel = remap[image_index] >= 0
            sub.append((remap[image_index[sel]].astype(np.int32), category[sel], score[sel], boxes[sel]))
        fused = fuse(sub, weights, sub_images, 0.6)
        rescore(fused, sub_images, 2, 0.15, 0.5, inplace=True)
        fused, _ = normalize_predictions(fused, sub_images)
        grouped = {image["id"]: [] for image in sub_images}
        for p in fused:
            values = [p["category_id"] - 1, p["score"], *p["bbox"]]
            grouped[p["image_id"]].extend([str(values[0]), *(f"{v:.12g}" for v in values[1:])])
        writer.writerows((image["id"], " ".join(grouped[image["id"]])) for image in sub_images)
        del fused, sub, grouped; gc.collect()
del parts; gc.collect()
check = submit.validate_submission(images, out / name)
write_json(out / (name.replace(".csv", "") + "_info.json"), {"test": files, "weights": weights, "iou": 0.6,
           "boxboost": {"k": 2, "alpha": 0.15, "thr": 0.5}, "per_clip": True, "check": check})
print(name, check, flush=True)
