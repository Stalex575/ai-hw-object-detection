# Experiments: first baselines

These are the first four runs, trained overnight on an RTX 5060 Laptop GPU (8 GB, WSL2). All of them use the same protocol: fold-0 clips, every 5th frame, masked training images, horizontal flip + scale jitter, seed 42, AMP, and a fixed number of epochs per run. All mAP numbers below come from the **shared evaluator** (`src/evaluate.py`: COCO mAP@[.5:.95], global top-500 per image, crowd/ignore regions from the supplied GT) on the 1,625 **original** fold-0 validation frames. The official `val/` set has not been touched.

## Summary

| Run | Input | Epochs | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | FPS* | Train time |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26s (smoke) | 640 | 20 | 0.187 | 0.343 | 0.183 | 0.041 | 0.129 | 0.308 | 51.4 | 0.41 h |
| YOLO26s | 1280 | 25 | 0.237 | 0.453 | 0.221 | **0.075** | 0.193 | 0.329 | 40.2 | 1.59 h |
| RT-DETR-L | 1280 | 12 | 0.179 | 0.324 | 0.175 | 0.046 | 0.145 | 0.262 | 19.5 | 2.26 h |
| **Faster R-CNN R50-FPN v2** | 1280** | 12 | **0.272** | **0.515** | **0.255** | 0.064 | **0.212** | **0.415** | 16.3 | 1.65 h |

\* Batch 1, FP16, including JPEG decoding, preprocessing, inference and post-processing, measured on the RTX 5060 Laptop (the plan specifies a T4, so these numbers are only comparable with each other).
\** Long side 1280 (short side 720) at test time; during training the short side is jittered between 544 and 896.

All four runs produced a valid `submission.csv` (3,811 rows). None has been uploaded to Kaggle yet. The machine-readable table is [experiments.csv](experiments.csv).

## Validation mAP during training

![Validation mAP per epoch](figures/val_map_per_epoch.png)

The two panels use **different evaluators** and should not be compared with each other. Ultralytics' built-in validator scores detections inside ignore regions as false positives, so it reads lower than the shared evaluator (YOLO26s 1280: 0.204 vs 0.237).

- **Both YOLO26s runs peak early and then slowly decline**: the smoke run peaks at epoch 1 of 20, and YOLO26s 1280 at epoch 5 of 25. So the COCO-pretrained model loses accuracy under this recipe. The likely causes are the auto-chosen AdamW (lr 0.00083) together with mosaic and the other default augmentations being switched off. This is worth a focused YOLO experiment.
- **RT-DETR-L is still improving steeply at epoch 12**, so it has not converged in this budget. That was expected, and the plan says to report it as such.
- **Faster R-CNN plateaus after epoch 6** (0.272 at epoch 6, 0.268–0.272 afterwards), so 12 epochs is enough for it.

> **Caveat: checkpoint selection differs between families.** The Ultralytics notebook evaluates `best.pt`, which Ultralytics picks by val mAP **on the same fold-0 frames**. The YOLO and RT-DETR numbers are therefore slightly optimistic (YOLO26s 1280 is reported from epoch 5). Faster R-CNN reports its **last** epoch. Before the final comparison, either evaluate `last.pt` for the Ultralytics runs or use the same rule for every family.

## Per-class mAP

![Per-class mAP](figures/per_class_map.png)

Faster R-CNN leads on every class except person and car, where it is roughly tied with YOLO26s 1280. Its biggest margins are on the large, rare vehicle classes: bus (0.49 vs 0.41), truck (0.255 vs 0.165) and van (0.26 vs 0.18). Bicycle is the weakest class for every model (≤ 0.075).

## Validation examples

The first frame of four fold-0 validation clips. Green boxes are ground truth; red boxes are predictions with score ≥ 0.1.

**YOLO26s 640 (smoke)**

![YOLO26s 640 validation examples](yolo26s_smoke_fold0_640_s42_v1/validation_examples.jpg)

**YOLO26s 1280**

![YOLO26s 1280 validation examples](yolo26s_base_fold0_1280_e25_s42_v1/validation_examples.jpg)

**RT-DETR-L 1280**

![RT-DETR-L validation examples](rtdetrl_base_fold0_1280_e12_s42_v1/validation_examples.jpg)

**Faster R-CNN 1280**

![Faster R-CNN validation examples](frcnn_base_fold0_1280_e12_b2_s42_v1/validation_examples.jpg)

## Current pick

**Faster R-CNN R50-FPN v2 at 1280 is the current best model**: it has the highest mAP (+0.035 over YOLO26s 1280), the best AP50/AP75 and the best medium and large AP. That holds even though it is reported from its last epoch and YOLO from its best one.

**YOLO26s 1280 is the runner-up and should stay in play**:

- it has the best AP_small, and small objects dominate this dataset;
- it is 2.5× faster;
- its own training curve suggests the recipe, not the model, is holding it back.

For the leaderboard, submit Faster R-CNN first and YOLO26s 1280 second. Because the two models are strong on different object sizes, they are natural candidates for the planned WBF ensemble.

These are single-fold, untuned baselines. Treat the ranking as provisional until the tuning experiments and the checkpoint-selection fix above are done.

## What is where

| Path | Content |
|---|---|
| `experiments/<run>/metrics.json` | Shared-evaluator metrics, including per-class AP and detection counts |
| `experiments/<run>/experiment.csv` | One summary row; all rows are combined in `experiments/experiments.csv` |
| `experiments/<run>/results.csv`, `results.png` | Ultralytics per-epoch training and validation curves (YOLO / RT-DETR) |
| `experiments/<run>/train_log.csv` | Per-epoch losses + shared-evaluator mAP every 2 epochs (Faster R-CNN) |
| `experiments/<run>/args.yaml`, `config.json` | Exact training settings |
| `experiments/<run>/runtime.json` | Library versions and git commit |
| `notebooks/runs/*.ipynb` | Executed run copies of `notebooks/04_anchor_free_yolo.ipynb` (only the run name, config and model class changed). `yolo26s_smoke…_v1.ipynb` stops with an error at the export cell (a zero-area box, since fixed in `src/yolo.py`); `…_post.ipynb` is its rerun from the saved weights. |
| `experiments/make_summary.py` | Rebuilds this folder (files, figures, Faster R-CNN examples) from `outputs/` |

Weights, predictions and submissions stay in the git-ignored `outputs/<run>/`. The `runtime.json` files record commit `eaea03e`, but the runs used the uncommitted versions of `src/tv_train.py`, `src/yolo.py` and the two new configs, which this commit adds.

### Reproduce

```bash
# Ultralytics runs: a run copy of notebooks/04 with RUN_NAME and the config path set, e.g. configs/yolo26s_base1280.yaml
# Faster R-CNN:
python -m src.tv_train --arch fasterrcnn --run-name frcnn_base_fold0_1280_e12_b2_s42_v1 --epochs 12 --batch 2 --eval-every 2 --test
python experiments/make_summary.py
```
