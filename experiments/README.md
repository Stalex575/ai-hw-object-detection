# Experiments

What failed and how it was fixed: see [Failures, fixes and retries](#failures-fixes-and-retries-lab-notebook).

All experiments ran on an RTX 5060 Laptop GPU (8 GB, WSL2), except Oleh's NWD run (Mac). All mAP numbers come from the **shared evaluator** (`src/evaluate.py`: COCO mAP@[.5:.95], global top-500 per image, crowd/ignore regions from the supplied GT). Tuning used fold-0 (1,625 original validation frames of the training clips); the final models were checked on the untouched official `val/` split, which turned out to track the leaderboard much better (see [Leaderboard check](#leaderboard-check-and-the-final-model-sun-nov-8)).

## Final results (Sun Nov 8)

| Submission | Official `val/` mAP | Public LB |
|---|---|---|
| Competition baseline | – | 0.221 |
| Faster R-CNN, fold-0 model (first submission) | 0.209 | 0.213 |
| Faster R-CNN, all training clips + low-res augmentation | 0.223 | 0.247 |
| WBF ensemble, 4 members (Faster R-CNN + YOLO26s, each plain + mirrored) | 0.255 | 0.270 |
| WBF ensemble, 8 members (+ one earlier checkpoint of each model) | 0.267 | 0.289 |
| **8-member ensemble + temporal rescoring** | 0.268 | **0.292** |

The official `val/` column for the ensembles is on all 2,846 frames (every 5th frame for the single models; the two agree within 0.001). Every step that helped is simple: more training data (all clips instead of one fold), a second, different detector, mirrored-image test-time augmentation, an extra checkpoint per model, Weighted Boxes Fusion, and boosting boxes that also appear in neighbouring video frames. Details, including what did *not* work, are in [Leaderboard check and the final model](#leaderboard-check-and-the-final-model-sun-nov-8); the fold-0 tables below are the tuning history that led there.

## Fold-0 baselines (Sat Nov 7, every run on its last epoch)

Team protocol rule (agreed Sat Nov 7): every run is reported on its **last** epoch, because Ultralytics picks `best.pt` on the same fold-0 frames we report on. The three Ultralytics runs were re-scored on `last.pt` (`tools/queue/reeval_last.py`, outputs in `outputs/<run>/last/`); the first table further down still shows their optimistic `best.pt` numbers.

| Run | Input | Epochs | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | FPS* | Train time |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26s (smoke) | 640 | 20 | 0.168 | 0.323 | 0.156 | 0.048 | 0.130 | 0.280 | 61.6 | 0.41 h |
| YOLO26s | 1280 | 25 | 0.216 | 0.413 | 0.200 | **0.070** | 0.174 | 0.324 | 45.4 | 1.59 h |
| RT-DETR-L | 1280 | 12 | 0.179 | 0.324 | 0.175 | 0.046 | 0.145 | 0.262 | 20.0 | 2.26 h |
| **Faster R-CNN R50-FPN v2** | 1280 | 12 | **0.272** | **0.515** | **0.255** | 0.064 | **0.212** | **0.415** | 16.3 | 1.65 h |
| RetinaNet R50-FPN v2 | 1280 | 12 | 0.231 | 0.434 | 0.217 | 0.042 | 0.175 | 0.376 | 21.0 | ~1.35 h** |

\* Same FPS protocol as below; the Ultralytics FPS were re-measured with `last.pt` (another day, so ±10 %).
\** WSL crashed during epoch 3; the run resumed from the epoch-2 checkpoint (`--resume`, same seed and data order). `experiment.csv` only counts the 1.08 h after the resume.

![All fold-0 runs](figures/fold0_runs_map.png)

- **Re-scoring on `last.pt` costs YOLO26s 1280 two points** (0.237 → 0.216) and the smoke run 1.9 points; RT-DETR-L is unchanged (its best epoch is its last). Faster R-CNN now leads YOLO26s by 5.6 points instead of 3.5.
- **RetinaNet baseline: 0.231.** It sits between YOLO26s and Faster R-CNN and is still rising slowly at epoch 12 (0.222 → 0.226 → 0.229 → 0.231 at epochs 6/8/10/12). Its weakest point is small objects (AP_small 0.042 vs 0.064 for Faster R-CNN). This is expected from its default anchors, see below. Settings: SGD lr 0.00125 (torchvision's RetinaNet reference 0.01 @ batch 16, scaled to batch 2), otherwise the Faster R-CNN recipe.
- **Bug found and fixed while starting RetinaNet:** torchvision's RetinaNet head keeps a logit for label 0, which never gets a positive, but early in training it still emitted label-0 boxes above the 0.001 score threshold and the shared evaluator rejected them (`category_id must be in 1..8`). `tv_train.predict` now drops label 0.

## Single-knob experiments (L2–L7, D1)

| Run | Change vs family baseline | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | Δ mAP |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN baseline | – | 0.272 | 0.515 | 0.255 | 0.064 | 0.212 | 0.415 | |
| Faster R-CNN baseline, seed 43 `frcnn_base…_s43…` | seed 42 → 43 (noise check, no change) | 0.273 | 0.521 | 0.253 | 0.068 | 0.211 | 0.419 | +0.000 |
| L2 `frcnn_anchors16-256…` | RPN anchor sizes 32–512 → 16–256 | 0.266 | 0.517 | 0.241 | **0.073** | **0.216** | 0.399 | −0.006 |
| L3 `frcnn_giou…` | RoI box loss smooth-L1 → GIoU (weight 10) | **0.275** | **0.522** | 0.255 | 0.062 | 0.209 | **0.418** | +0.003 |
| RetinaNet baseline | – | 0.231 | 0.434 | 0.217 | 0.042 | 0.175 | 0.376 | |
| L4 `retinanet_kmeans…` | anchors → per-level k-means (9 shapes per level) | 0.170 | 0.388 | 0.125 | 0.060 | 0.164 | 0.233 | **−0.061** |
| L5 `retinanet_gamma1.5…` | focal loss γ 2.0 → 1.5 | 0.232 | 0.436 | 0.217 | 0.042 | 0.180 | 0.379 | +0.001 |
| L6 `frcnn_posfrac0.5…` | RoI sampler positive fraction 0.25 → 0.5 | 0.274 | 0.519 | 0.255 | 0.066 | 0.206 | **0.422** | +0.002 |
| L7 `retinanet_iou0.4-0.3…` | anchor matcher IoU fg/bg 0.5/0.4 → 0.4/0.3 | 0.201 | 0.369 | 0.196 | 0.022 | 0.141 | 0.343 | **−0.030** |
| L4b `retinanet_kmeansshared…` | anchors → shared (scale-equivariant) k-means, base 16–256 | 0.200 | 0.421 | 0.166 | **0.060** | 0.162 | 0.284 | **−0.031** |
| L6, seed 43 `frcnn_posfrac0.5…_s43…` | L6 replicate | 0.267 | 0.510 | 0.247 | 0.065 | 0.208 | 0.425 | −0.006 (vs seed-43 baseline) |
| L3+L6 `frcnn_giou_posfrac0.5…` | GIoU RoI loss **and** positive fraction 0.5 (the two knobs that did not hurt) | 0.268 | 0.512 | 0.250 | 0.064 | 0.202 | 0.420 | −0.004 |
| YOLO26s baseline (last epoch) | – | 0.216 | 0.413 | 0.200 | 0.070 | 0.174 | 0.324 | |
| Y2 `yolo26s_nwd…_mac_v1` (Oleh, MacBook) | box loss CIoU → 0.5·CIoU + 0.5·NWD | 0.221 | 0.413 | 0.206 | 0.066 | 0.176 | **0.352** | +0.005 |
| D1 `frcnn_nomask…` | Faster R-CNN trained on the **original** frames instead of the ignore-region-masked ones | 0.266 | 0.499 | 0.251 | 0.059 | 0.200 | 0.413 | −0.007 |

- **L2:** halving the RPN anchors trades large for small objects. AP_small +1.0 (best of all runs so far), AP50 +0.3, person/car/bicycle slightly up; but AP_large −1.6 (bus 0.49 → 0.47, truck 0.255 → 0.226) and AP75 −1.3, so the net mAP is −0.6. The two-stage head already recovers most small boxes from the low-quality RPN matches, so the anchor-coverage gain (71.7 % → 94.5 %) translates into much less than it suggests. A cheaper variant for later: keep 32–512 and add a 16 px size on P2/P3 instead of shifting every level.

- **L3:** GIoU on the RoI head (both deltas decoded against a unit box, which equals GIoU on the real boxes; weight 10 as in mmdetection) gives +0.3 mAP / +0.7 AP50, but almost all of it comes from the two rare large classes (bus +1.6, truck +1.5 mAP); the frequent classes move by ≤0.5 in either direction and AP75 is unchanged. So it does not buy the localisation gain it targets, and +0.3 is within what a different seed could give. Same speed (16.4 FPS) and train time (1.62 h). It is a free change, so it can be kept for the final Faster R-CNN, but it is not evidence by itself. TIDE on L3 (`experiments/tide_frcnn_giou/`) agrees: the Loc error impact only drops 5.1 → 4.8 AP50 points; Cls 13.4 → 12.3, Miss 6.3 → 7.0, the rest unchanged.

- **L4 failed, and the reason is a design error in the per-level anchors, not k-means as such.** AP_small rises 4.2 → 6.0 and bicycle/motor improve slightly, but AP75 drops 21.7 → 12.5 and AP_large 37.6 → 23.3 (bus 0.41 → 0.20, truck 0.20 → 0.11, van 0.24 → 0.14). It started far behind (0.107 vs 0.179 at epoch 2) and never caught up. RetinaNet's classification and box heads are **shared across P3–P7**: anchor slot *j* is the same set of conv weights on every level, so torchvision gives slot *j* the same shape relative to the stride on every level. Fitting 9 shapes independently per level broke that: slot 0 is 7×13 (tall) on P3, 33×30 (square) on P4, 161×92 (wide) on P6, so one set of weights has to regress different shapes, and the pretrained per-slot weights start from the wrong shape. Localisation quality (AP75) and the large classes suffer most, which fits. **L4b** (queued as job 72, after L6/L7) fixes this with *scale-equivariant* k-means: one set of 9 shapes fitted on boxes divided by their level's base size, placed in the pretrained slots by max IoU, times the level base (16–256) on each level. Same coverage as per-level k-means (95.0 % vs 94.9 %).

- **L5:** a softer focal loss (γ 1.5, so easy negatives are down-weighted less) changes nothing: +0.1 mAP, every class within ±0.4 except bus +1.3, same AP_small. Same speed (20.7 FPS) and train time (1.31 h). Null result; the RetinaNet gap to Faster R-CNN (4 points) is not about the classification loss.

- **L6:** twice as many positive RoIs per image in the box-head batch gives +0.2 mAP: AP_large +0.7, AP_small +0.2, AP_medium −0.6. Almost all of it is van (0.261 → 0.285); every other class moves by ≤0.7. Van is the class most confused with car (TIDE), so more positives may help the head separate them, but one class on one seed is not enough to claim it. Same speed (16.7 FPS), 1.62 h. Null to slightly positive; it could be combined with L3 in the final recipe.

- **L7:** lowering RetinaNet's anchor-matching thresholds (positive at IoU ≥ 0.4 instead of 0.5, negative below 0.3 instead of 0.4) was meant to give small boxes more positive anchors. It did the opposite: −3.0 mAP, and the loss is concentrated on small objects (AP_small 4.2 → 2.2; van and bicycle small AP fall to 0) and on AP50 (43.4 → 36.9), while bus improves (0.41 → 0.43). Our reading: a lower threshold adds many more positive anchors to large objects than to small ones, and the focal loss is normalised by the number of positives, so the gradient from small objects is diluted; the extra, poorly aligned positives also make the scores noisier. The pretrained head was also calibrated for 0.5/0.4. Same speed (20.8 FPS), ~1.38 h in the queue. Negative result; keep 0.5/0.4.

- **L4b (the retry of L4) recovers half of L4's loss but is still −3.1 mAP below the baseline.** Making the anchors scale-equivariant fixed what we diagnosed: 0.170 → 0.200, AP75 0.125 → 0.166, AP_large 0.233 → 0.284, bus 0.20 → 0.34. AP_small keeps L4's gain (4.2 → 6.0; car small 0.12 → 0.22) and bicycle/motor are the best of any RetinaNet run (0.057 / 0.098). But large objects are still well below the baseline (AP_large 0.376 → 0.284, truck 0.20 → 0.13, van 0.24 → 0.16), and the curve flattens after epoch 8 (0.201, 0.201, 0.200) while the baseline was still rising. The likely reason is the same trade-off as L2: the k-means anchors are centred on **bases 16–256 instead of 32–512**, so the largest anchor is about 410 px instead of 810 px, and large buses and trucks at 1280 px are matched only through the low-quality fallback. So L4b changes two things at once (shapes and scale). The clean follow-up would be the same 9 shared shapes on bases 32–512, but with RetinaNet 4 points behind Faster R-CNN it is not worth the GPU time before the final runs. Same speed (20.6 FPS), ~1.38 h.

- **D1 (data ablation): masking the ignore regions in the training frames helps a little.** Our preprocessing fills the dataset's ignore regions (crowds, far-away parked vehicles that are not labelled) with the image's mean colour before training (`src/convert.py`). Without it, those unlabelled objects are background for the loss, so the model is trained to suppress things that look exactly like labelled objects. Trained on the original frames, Faster R-CNN drops 0.272 → 0.266 (AP50 −1.6, AP_small −0.5). The loss is in person (0.210 → 0.195, AP50 0.562 → 0.520), truck (0.255 → 0.219) and bicycle/motor (−0.6/−0.8). Bus goes up (0.488 → 0.508) and car is unchanged. The unmasked run is ahead at epoch 2 (0.259 vs 0.254), when it has seen more context, and falls behind from epoch 4. The validation frames are the original ones in both cases; the evaluator ignores detections inside ignore regions, so masking changes only the training signal. The seed-43 replicate (below) moves person by 0.1 and truck by 0.6, so the person and truck losses (−1.5 / −3.6) are well above seed noise. Same speed (16.7 FPS), ~1.70 h in the queue. Masking stays in the pipeline.

### How much is seed noise? (Faster R-CNN baseline, seed 42 vs 43)

The same baseline run with seed 43 (different weight init of the new layers, data order and augmentation) gives **0.2728 vs 0.2725**: overall mAP moves by only 0.03 points, AP50 by 0.6, AP_small by 0.4. Per class the picture is different: the frequent classes barely move (person 0.1, car 0.3, van 0.2 points), but the rare ones do (**bus 1.6**, bicycle 1.1, motor 0.9, tricycle 0.5, truck 0.6). Mid-training numbers are noisier still: at epoch 4 the two seeds are 0.258 and 0.270. This is one pair of seeds, so it is a rough estimate of the noise, not a variance, but it is enough to re-read the single-knob results:

| Run | Δ mAP | Class changes larger than the seed difference | Verdict |
|---|---|---|---|
| L2 anchors 16–256 | −0.6 | truck −2.9, tricycle −1.3 | Real, negative. |
| L3 GIoU | +0.3 | none clearly: bus +1.6 equals bus's seed difference, truck +1.5 vs 0.6 | **Not distinguishable from noise.** |
| L6 positive fraction 0.5 | +0.2 | van +2.4 (seed difference 0.2) | **Not real:** with seed 43, L6 gives 0.267 and van 0.258. Two seeds of the baseline agreeing closely understated the noise. |
| D1 unmasked frames | −0.7 | person −1.4, truck −3.6; bus +2.0 (seed 1.6) | Real, negative: masking helps. |

So the overall mAP of one run is reproducible to well under 0.1 points here, while single rare-class APs need a change of about 2 points before we read anything into them.

**L3 + L6 combined (the candidate for the final Faster R-CNN) is worse than either alone: 0.268, −0.4 vs the baseline.** The two small gains do not add up. L6's van gain mostly disappears (0.285 alone → 0.266 combined), and truck drops to 0.228 (−2.7) although GIoU alone had raised it to 0.270. The curve rises more slowly from the start (0.252 / 0.255 / 0.261 at epochs 2/4/6 vs 0.254 / 0.258 / 0.272) and is still rising at epoch 12, so it may simply need a different schedule, but we did not test that. Truck AP across the six Faster R-CNN runs ranges from 0.219 to 0.270, much wider than the 0.6-point difference between the two seeds, so one seed pair **underestimates** the noise of rare classes. Our conclusion: none of the single knobs gives a gain we can trust, so the final Faster R-CNN keeps the **baseline settings**. **L6 with seed 43 settles it: 0.267, and van is back at 0.258** (0.285 with seed 42, 0.263 for the seed-43 baseline). The van gain did not repeat, so L6 is noise too. Same speed (16.6 FPS), ~1.74 h in the queue.

- **Y2 (Oleh, trained on his MacBook):** mixing the Normalized Gaussian Wasserstein Distance into YOLO26s's box loss (α = 0.5, C = 12.8) gives +0.5 mAP on the last epoch. NWD is designed for tiny objects, but the gain is where we did not expect it: AP_large +2.8 (bus +1.6, van +1.4), while AP_small goes slightly **down** (7.0 → 6.6) and AP50 is unchanged. Like the baseline, the Ultralytics validation curve peaks early and declines (0.229 at epoch 1 → 0.206 at epoch 25). Not a clean single-knob comparison: it ran on Apple MPS instead of CUDA and dropped from batch 4 to batch 2 after an out-of-memory restart at epoch 8 (see the lab notebook). With rare-class noise of 1–2 points, +0.5 overall is not enough to change YOLO's place behind Faster R-CNN (0.272).

## Leaderboard check and the final model (Sun Nov 8)

| Submission | fold-0 val | official `val/`* | public LB |
|---|---|---|---|
| Faster R-CNN baseline (fold-0 model), inference at short side 720 | 0.272 | **0.209** | **0.213** |
| same model, inference at short side 1080 | 0.279 | – | 0.200 |
| **Final model**, all training clips + low-res augmentation, last epoch (6) | – | **0.223** | **0.247** |
| final model, epoch-2 checkpoint (`best.pt`) | – | 0.227 | – |
| YOLO26s, all training clips, Ultralytics default augmentations, last epoch (12) / epoch 4 | – | 0.194 / 0.228 | – |
| **WBF ensemble, 4 members**: Faster R-CNN (last) + YOLO (epoch 4), each plain + mirrored | – | **0.255** | **0.270** |
| WBF ensemble, 8 members: the 4 above + Faster R-CNN epoch 2 + YOLO epoch 3, each plain + mirrored | – | 0.267 | **0.289** |
| 8-member ensemble + temporal rescoring (`src/temporal.py`, ±2 frames, alpha 0.15, IoU 0.5) | – | 0.268 | **0.292** |

\* Every 5th frame for single models, all 2,846 frames for the ensembles (the two agree within 0.001).

- **Final YOLO** (`yolo26s_final_all_step2_e12_b3_s42_v1`, `src/yolo_final.py`): YOLO26s 1280 on the same 12,114 frames as the final Faster R-CNN, this time with the Ultralytics default augmentations (mosaic, HSV, translate, scale) and SGD lr 0.01, 12 epochs, batch 3. **It again peaked early and then declined** (Ultralytics val mAP 0.228 at epoch 4 → 0.194 at epoch 12, our evaluator 0.194 at the last epoch), so turning the augmentations back on did not fix the early peak. For the ensemble we used the epoch-4 checkpoint (0.227 on all official val frames), chosen on the official val split; this makes the val numbers below slightly optimistic, the LB is the honest check.
- **WBF ensemble** (`src/ensemble.py`, Weighted Boxes Fusion, per image and class, IoU 0.6, equal weights): each model also predicts on the horizontally mirrored frame (`--hflip`, boxes mirrored back), and all prediction sets are fused. On all 2,846 official val frames: Faster R-CNN alone 0.222, YOLO alone 0.227, either model + its mirrored pass 0.239 / 0.234, both models without mirroring 0.241, **all four 0.255 (LB 0.270, +2.3 over the single model)**. Weighting one model 2:1 was worse than equal weights (0.251); the IoU threshold barely mattered (0.253–0.256 for 0.5–0.7). Adding one more checkpoint of each model (Faster R-CNN epoch 2, YOLO epoch 3, plain + mirrored) gave 0.262 each and 0.267 with both.
- **Temporal rescoring** (blend each box's score with the best-overlapping same-class box in the ±k neighbouring frames, 0 if none): on every-5th-frame val it hurt (−0.3 to −1.5; objects move too far in 5 frames). On consecutive frames, on top of the 8-member ensemble: +0.15 at best (0.2668 → 0.2683, k=2, alpha 0.15, IoU 0.5), and −0.3 to −0.4 with IoU 0.3. **Within noise on val**, yet on the public LB it added +0.3 (0.2890 → 0.2919). The test clips are all consecutive frames, while our val gain was measured on only 7 clips, so the LB (3,811 frames, 48 clips) is the better estimate here; we report it as a small, consistent but unconfirmed gain.

- **Fold-0 validation over-estimates the leaderboard by ~6 points; the untouched official `val/` split does not** (0.209 vs 0.213). About 1 point is ignore-region handling: the test set ships no ignore regions, and re-scoring fold-0 val without them gives 0.262 instead of 0.272. The rest is mostly frame size: fold-0 val frames are big (1904×1071, 2688×1512), test frames are small (1360×764 and 1920×1080 mostly, plus 600 frames at 680×382 / 960×540, a size that does not occur in training at all). From here on, decisions use the official `val/` split.
- **Inference at a larger size helps only frames that would otherwise be shrunk.** Per frame size on fold-0 val (720 → 1080): 2688×1512 0.297 → 0.313, 1904×1071 0.258 → 0.257, **1344×756 0.289 → 0.281**. Test frames are small, so a larger inference size mostly enlarges them, which matches the LB drop to 0.200. For the same reason we did **not** do tiling/SAHI: slicing small frames into tiles and enlarging them is the same operation.
- **Final model** (`frcnn_final_all_step2_lowres_e6_b2_s42_v1`): the unchanged baseline recipe (720 px, masked frames, no single-knob changes, since none survived the seed check), trained on **all 56 training clips, every 2nd frame** (12,114 frames vs 3,243 for the fold-0 model) for 6 epochs, plus a **low-resolution augmentation** (`--lowres-p 0.3`: shrink the frame to 640–1000 px wide and enlarge it back, so the model also sees blurry low-resolution objects like the 680×382 / 960×540 test clips). Scored on the official `val/` split.
- **Result:** official `val/` mAP **0.223** at the last epoch vs 0.209 for the submitted fold-0 model (+1.4). Per epoch: 0.227 (epoch 2), 0.224 (4), 0.223 (6), so the extra epochs did not help overall, but **AP_small kept rising** (0.106 at epoch 2 → 0.115 at epoch 6, vs 0.095 for the fold-0 model). Per class vs the fold-0 model: tricycle +6.1, van +3.8, car +1.9, truck +1.9, person +1.5, bus +0.7, motor −1.0, bicycle −3.5. Training took 2.69 h (6,057 it/epoch, 4.5 GB VRAM).
- Two submissions written: the last epoch (`submission.csv`, our reported model, same fixed-length protocol as every other run) and the epoch-2 checkpoint (`predict_test_epoch2/submission.csv`). The 0.4-point gap between them on `val/` is within seed noise, so the last epoch is the primary submission.

## Anchors vs object size (A3)

`src/anchors_kmeans.py` measures, on 400 fold-0 training frames at the training scale (short side 720), the best IoU between each GT box and **any anchor placed where torchvision places it** (centres on each level's stride grid). A GT whose best IoU is < 0.5 only gets a positive anchor through the low-quality-match fallback.

| Anchor set | mean best IoU | GT with best IoU ≥ 0.5 | <16 px | 16–32 px | 32–64 px | ≥64 px |
|---|---|---|---|---|---|---|
| torchvision default, sizes 32–512 | 0.605 | 71.7 % | **0 %** | 60 % | 100 % | 100 % |
| sizes 16–256 (same ratios) | 0.657 | 94.5 % | 56 % | 99 % | 99 % | 99 % |
| k-means, 15 shapes split by area over P3–P7 | 0.625 | 86.5 % | 63 % | 96 % | 92 % | **47 %** |
| k-means per level (L4) (boxes → level by size, 9 shapes per level) | **0.660** | 94.9 % | 62 % | 99 % | 99 % | 98 % |
| **k-means shared across levels** (L4b: boxes ÷ level base, one set of 9 shapes × base 16–256) | 0.655 | **95.0 %** | 62 % | 99 % | 99 % | 99 % |

(GT counts per bucket: 2,088 / 8,249 / 6,744 / 1,861. Median box at training scale: 30 px.)

![Anchor best-IoU histogram](anchors_kmeans/anchor_best_iou_hist.png)

- With the default anchors, **no box under 16 px and 40 % of the 16–32 px boxes ever reach IoU 0.5 with an anchor**. That matches RetinaNet's weak AP_small and is the motivation for L2 (Faster R-CNN RPN anchors 16–256) and L4 (RetinaNet k-means anchors).
- **Plain k-means is worse than simply halving the sizes**: splitting 15 clusters over the levels by area puts mid-size boxes on coarse levels (P6/P7, stride 64/128), where the anchor grid is too sparse to reach IoU 0.5. The stride-aware variant (assign boxes to a level first, then cluster within the level) fixes that and is what L4 uses (`experiments/anchors_kmeans/anchors_kmeans_per_level.json`; P7 has only 33 boxes, so it keeps the torchvision layout at base 256).

## Error analysis of Faster R-CNN (A6, TIDE-style)

`src/tide.py` assigns every non-TP detection one error type at IoU 0.5 (TIDE definitions, t_b = 0.1) and measures how much **AP50** would rise if only that error type were fixed, with the shared evaluator (baseline AP50 0.515).

| Error | Cls | Loc | Both | Dup | Bkg | Miss | all FP | all FN |
|---|---|---|---|---|---|---|---|---|
| AP50 gain if fixed (points) | **+13.4** | +5.1 | +1.1 | +0.0 | +3.4 | +6.3 | +24.5 | +14.2 |

![TIDE](tide_frcnn/tide.png)

- **Class confusion is the largest error, by far.** The most frequent confusions (score-weighted): car ↔ van (both directions), motor → bicycle, person → motor / bicycle (riders), motor ↔ tricycle, truck → bus. These are visually similar vehicles at 20–40 px, plus rider-vs-vehicle boxes that overlap almost perfectly.
- **Misses (+6.3) come next.** In the failure frames below they are mostly small pedestrians and parked two-wheelers along the sidewalks of dense scenes.
- **Duplicates don't matter** (+0.0), so a looser NMS is unlikely to hurt, but class-agnostic NMS could; it would also remove real rider + motor pairs.
- Implications: class-aware ensembling (WBF) and per-class score calibration are more promising than more localisation work; small-object recall is the other lever (anchors, higher resolution, SAHI).

Failure cases (yellow: missed GT, green: TP, red: background FP, orange: localisation, magenta: wrong class, cyan: duplicate; detections with score ≥ 0.3):

![Failure 1](tide_frcnn/failure_1.jpg)
![Failure 2](tide_frcnn/failure_2.jpg)
![Failure 3](tide_frcnn/failure_3.jpg)

## Failures, fixes and retries (lab notebook)

Every run or step that failed, what caused it, and what was changed before the retry. Times are Kyiv time. The raw evidence is in `outputs/queue/queue.log` (START/END with exit codes) and `outputs/queue/state/<job>.log`. **Rule from now on: each new failure gets a row here, including ones that are our own mistakes.**

### Runs and jobs that failed

| When | Run / job | What happened | Cause | Fix, then retry | Outcome |
|---|---|---|---|---|---|
| Fri 22:23–22:49 | `10_yolo26s_smoke640` (YOLO26s 640 smoke) | Training finished, but the export of the validation predictions crashed: `ValueError: Prediction 498: width and height must be positive` (exit 1). | Ultralytics returned zero-area boxes (542 of 746k predictions). The shared evaluator rejects them. | `src/yolo.py` drops zero-area boxes (on `main`, commit `5c84136`). Only the post-processing was re-run (job `15_…_post`, 3 min); no retraining. | Smoke run scored (0.187 `best.pt`, 0.168 `last.pt`). The same fix protected the later YOLO and RT-DETR runs. |
| Fri | RT-DETR-L 1280, batch 4 | No error, but training was ~10× slower than expected. | The GPU ran out of its 8 GB, and the Windows driver silently spilled into shared system RAM instead of raising an out-of-memory error. | Batch 2 (`configs/rtdetr_l_base1280.yaml`). Healthy-run check added to the monitoring rules: < 7.6 GB VRAM and > 60 W. | Ran at full speed: 0.179, 2.26 h. |
| Fri | Faster R-CNN 1280, batch 4 | Out of memory in the batch probe. | Faster R-CNN at 1280 px needs more than 8 GB at batch 4. | `outputs/queue/frcnn_probe.py` caps memory at 82 % so an overflow raises an error instead of spilling, tries batch 4, then batch 2 (peak 5.3 GiB). | Batch 2 used for all torchvision runs; baseline 0.272. |
| Sat 10:12–10:18 | `50_retinanet_base1280` (L1), first attempt | Crash in the evaluator: `category_id must be in 1..8`. Job stopped (exit 143). | torchvision's RetinaNet head keeps a logit for label 0. It never gets a positive anchor, but early in training it still produced label-0 boxes above the 0.001 score threshold. | `tv_train.predict` drops label 0. Restarted from scratch at 10:19. | Fixed; also protects every later RetinaNet run. |
| Sat ~10:35 | L1 (RetinaNet baseline), second attempt | **WSL crashed** at epoch 3; all processes were killed. | **Our mistake:** we ran a test inference (`src/tv_predict.py`) next to the training process. WSL has ~10 GB RAM, and two torch processes plus data-loader workers exceeded it. | After the user restarted WSL, the job resumed from the epoch-2 checkpoint (`--resume`, same seed and data order) at 10:49. New rule: never run a second torch process during training; tests go into the queue. | Finished: 0.231. `experiment.csv` counts only the 1.08 h after the resume (real ≈ 1.35 h). |
| Sat ~14:50 | During L2 (FRCNN anchors 16–256) | No run failed, but available RAM fell below 1 GB during the validation pass (2.4 GB swap used), and the system killed a background monitoring process. | Each eval forks data-loader workers on top of the ~6 GB training process. | New flag `--eval-workers` (inference only); jobs L3+ use 2 workers for evals. | No memory problems in L3–L7 or L4b. |
| Sun (Oleh's MacBook) | Y2 `yolo26s_nwd_fold0_1280_e25_s42_mac_v1` | Out of memory on Apple MPS during epoch 8 at batch 4. | 1280 px YOLO26s at batch 4 exceeds the MPS memory limit on an 18 GB MacBook. | Restored the epoch-7 checkpoint and repeated epoch 8 onwards at batch 2. | Finished: 0.221, ~7.8 h in total (5.5 h after the restart). The batch change makes it a less clean comparison with the batch-4 baseline. |
| Sun ~13:20 | Leaderboard submission at inference size 1080 | Scored 0.200 on the public LB vs 0.213 at 720, although fold-0 val had gone up (0.272 → 0.279). | **Our mistake:** we judged a resolution change on fold-0 validation, whose frames are much larger than the test frames. On test-like 1344×756 frames, 1080 is worse (0.289 → 0.281). | Per-resolution evaluation; the official `val/` split (0.209 for the submitted model, LB 0.213) is now the reference. Tiling/SAHI dropped for the same reason. | One leaderboard submission spent; the lesson changed the final plan. |
| Sun 12:53 | 1080 px **training** smoke test (fold 0, batch 2) | `RuntimeError: CUDA driver error: device not ready` in the RPN anchor matching, 7.6 GB VRAM. | The GT × anchor IoU matrix at 1350×2400 px (scale jitter on top of 1080) does not fit in 8 GB. | 900 px fits (6.0 GB) and was queued, then paused once the leaderboard showed larger sizes hurt on test. | Not trained. |
| Sun 17:22 | `82_yolo26s_final_all_step2` (final YOLO, batch 4) | No error, but VRAM reached 7.7 GB of 8 within the first 230 iterations. | Mosaic packs four dense frames into one 1280 px input, so the label assigner sees several hundred boxes per image. Above ~7.6 GB the Windows driver silently spills into system RAM (see the RT-DETR row). | Killed after 2 min, restarted as job `83` at batch 3 with `expandable_segments` (Ultralytics still accumulates to an effective batch of 64). | 4.6 GB, 14 img/s. |
| Sun 21:10 | Ensemble / temporal analysis on CPU (not a queue job) | Claude Code's background shells (the temporal-rescoring grid and two queue monitors) were stopped because WSL ran low on RAM. | The grid held four full prediction sets for all 2,846 val frames as Python dicts while a GPU prediction job was also using RAM. | Waited for the GPU job to end; ran the analysis alone. | GPU jobs unaffected. |
| Sun 21:25 | 8-member ensemble grid | First under `ulimit -v`: segmentation fault (the cap counts virtual memory that threads reserve, not real use). Then without the cap: our watchdog killed it at 1.0 GB free RAM. | Eight prediction sets as Python dicts (~0.5 GB each). | `src/ensemble.py` now stores each member as numpy arrays (`compact()`, ~10x less RAM); reproduces the 4-member result exactly (0.2552). | Ran with ≥2.6 GB free. |
| Sat 15:49–17:12 | `54_retinanet_kmeans` (**L4**) | Ran to the end, but scored **0.170 vs 0.231** for the baseline (AP75 0.217 → 0.125, bus 0.41 → 0.20). | **Design error** in our anchors. RetinaNet's head is shared across P3–P7, so an anchor slot must mean the same shape on every level. Per-level k-means put unrelated shapes in the same slot (tall on P3, square on P4, wide on P6). Details in *Single-knob experiments*. | `levels_from_shared_kmeans` in `src/anchors_kmeans.py`: one set of 9 shapes on stride-normalised boxes, placed in the pretrained slots by max IoU. Same coverage (95.0 % vs 94.9 %). | Retry L4b (Sun 04:30–05:53): **0.200**. Half of the loss recovered, still −3.1 vs the baseline (large objects; see L4b above). |

### Ideas that did not help (negative or null results)

| Experiment | Expected | Got | Lesson |
|---|---|---|---|
| YOLO26s fine-tuning, 25 epochs (Fri) | Steady improvement | Ultralytics val mAP peaks at epoch 1 (640) or 5 (1280), then falls slowly. Re-scored on `last.pt` it loses 2 points vs `best.pt`. | Probably lr too high for fine-tuning and/or overfitting. Follow-ups Y1/Y2 (Oleh). |
| Checkpoint choice (Fri) | – | Ultralytics `best.pt` is selected on the same fold-0 frames we report on, so YOLO/RT-DETR looked 2 points better than they are. | Protocol rule: every run is reported on its last epoch. All three Ultralytics runs re-scored (L0). |
| A3: plain k-means anchors (15 shapes split by area) | Better coverage than the defaults | 86.5 % of boxes reach IoU 0.5, worse than simply halving the anchor sizes (94.5 %); only 47 % of boxes ≥ 64 px. | Coarse levels have sparse grids: anchors have to be assigned per level first. Not trained. |
| L2: Faster R-CNN anchors 16–256 | Better small objects | AP_small +1.0, but AP_large −1.6 → mAP −0.6. | Anchor coverage (72 → 95 %) mostly doesn't matter for a two-stage detector. |
| L3: GIoU box loss | Better localisation (AP75, TIDE Loc) | mAP +0.3 only from bus/truck, AP75 unchanged, TIDE Loc impact 5.1 → 4.8. | Null result; harmless to keep. |
| L4: per-level k-means anchors | Better small objects for RetinaNet | −6.1 mAP (see the table above). | Shared heads need scale-equivariant anchors → L4b. |
| L5: focal loss γ 1.5 | More signal from hard-ish examples | +0.1 mAP, all classes unchanged. | Null result; the loss shape is not RetinaNet's bottleneck. |
| L6: RoI positive fraction 0.5 | Better recall of small/rare objects | +0.2 mAP, almost all from van (+2.4). | Null to slightly positive; possibly helps car/van separation. |
| L7: RetinaNet matcher IoU 0.4/0.3 | More positive anchors for small boxes | −3.0 mAP; AP_small 4.2 → 2.2, AP50 −6.5. | Lower thresholds favour large objects (more positives) and dilute the small ones. |
| D1: train on unmasked frames | More visual context, and no artificial mean-colour patches | −0.7 mAP (person −1.5, truck −3.6; bus +2.0). | Unlabelled objects become false negatives in training; masking stays. Above seed noise (seed 43 moves person 0.1, truck 0.6). |
| Y2: NWD in the YOLO box loss (Oleh) | Better tiny-object recall | +0.5 mAP, but from large objects (AP_large +2.8); AP_small −0.4. | The tiny-object loss did not help tiny objects here; not enough to close the gap to Faster R-CNN. |
| L3 + L6 combined | Add the two small gains (+0.3, +0.2) | 0.268, −0.4: van gain gone (0.285 → 0.266), truck −2.7. | Small single-knob gains on one seed do not stack; the final Faster R-CNN keeps the baseline settings. |
| L6 seed 43 (check of the van gain) | Van +2.4 again | 0.267, van 0.258 (no gain). | The L6 gain was noise. Lesson: one replicate of the baseline is not enough to call a rare-class gain real. |
| Seed 43 replicate (noise check) | – | 0.2728 vs 0.2725; rare classes move up to 1.6 (bus). | L3's +0.3 (bus/truck) is within noise; L6's van gain and D1's losses are not. |
| L4b: shared k-means anchors (retry of L4) | Fix L4, keep its small-object gain | 0.200: +3.0 over L4, AP_small 6.0 kept, but −3.1 vs the baseline because AP_large is 0.284 vs 0.376. | Fixing the design error worked, but anchors fitted to the median box (30 px) shrink the scale range: large objects lose. Small-object gains cost large-object AP for both detectors (L2, L4b). |

### Mistakes in our own analysis tools (caught before any number was reported)

- **TIDE score floor:** the first version of `src/tide.py` dropped detections below 0.01 score, so its baseline AP50 was ~1 point below the shared evaluator's. Fixed: default `--min-score 0`, keeping all detections.
- **TIDE `--examples 0`** drew every clip instead of none. Fixed with an early return.
- **Low-resolution augmentation, first version:** our code rounded and clamped the image to 0–255, but the loader returns floats in [0, 1], so augmented frames would have become black-and-white. Caught by rendering one augmented frame before any training; fixed to clamp to [0, 1].
- **Data prep command:** `convert.py coco --out` expects a file name (`prep/coco/train_step2.json`), not a folder; the first attempt failed with `IsADirectoryError` and was re-run.
- **Memory cap too tight:** TIDE on L3 crashed with `MemoryError` under a 3.5 GB virtual-memory cap. It was re-run after L4's eval with a 7 GB cap.

## First baselines (Fri night; YOLO / RT-DETR from `best.pt`)

### Summary

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

The figure shows the four 1280 baselines on their **last** epoch. Faster R-CNN leads on every class; car is a tie with YOLO26s (0.495), and person is close (0.21 vs 0.20). Its biggest margins are on the large, rare vehicle classes: bus (0.49 vs 0.41 RetinaNet / 0.36 YOLO26s), truck (0.255 vs 0.20 / 0.13) and van (0.26 vs 0.24 / 0.17). RetinaNet is second on the vehicle classes but last-but-one on person (0.17) and motor (0.09), the small classes. Bicycle is the weakest class for every model (≤ 0.075).

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

**RetinaNet 1280**

![RetinaNet validation examples](retinanet_base_fold0_1280_e12_b2_s42_v1/validation_examples.jpg)

## Model choice

**Final pick (Sun Nov 8): the 8-member WBF ensemble of Faster R-CNN and YOLO26s with temporal rescoring (public LB 0.292).** See [Final results](#final-results-sun-nov-8).

How we got there (history, Sat Nov 7): on fold 0, Faster R-CNN R50-FPN v2 at 1280 was the best single model (+0.056 mAP over YOLO26s 1280 and +0.041 over RetinaNet, all on their last epoch), and YOLO26s 1280 was the runner-up with the best AP_small and 2.5× the speed. Because the two families make different errors, we planned a WBF ensemble of them from the start; once both were trained on all training clips they were about equally strong on the official `val/` split (0.223 vs 0.227), which is when the ensemble paid off (+3.2 mAP for 4 members).

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
| `experiments/<run>/last/` | Ultralytics runs re-scored on `last.pt` (the protocol numbers) |
| `experiments/anchors_kmeans/` | Anchor diagnostics and the k-means anchor files |
| `experiments/tide_frcnn/` | TIDE-style error breakdown + failure cases of Faster R-CNN |
| `experiments/final/` | Final models and ensembles: configs, metrics on the official `val/` split, Ultralytics curves of the final YOLO, ensemble and rescoring settings |
| `src/yolo_final.py` | Final YOLO: training on all clips, official-val scoring, plain + mirrored predictions |
| `src/ensemble.py`, `src/temporal.py` | Weighted Boxes Fusion and the neighbouring-frame rescoring |
| `experiments/make_summary.py` | Rebuilds this folder (files, figures, validation examples) from `outputs/`; extend `RUNS` for new runs |

Weights, predictions and submissions stay in the git-ignored `outputs/<run>/`. The `runtime.json` files record commit `eaea03e`, but the runs used the uncommitted versions of `src/tv_train.py`, `src/yolo.py` and the two new configs, which this commit adds.

### Reproduce

```bash
# Ultralytics runs: a run copy of notebooks/04 with RUN_NAME and the config path set, e.g. configs/yolo26s_base1280.yaml
# Faster R-CNN:
python -m src.tv_train --arch fasterrcnn --run-name frcnn_base_fold0_1280_e12_b2_s42_v1 --epochs 12 --batch 2 --eval-every 2 --test
python -m src.tv_train --arch retinanet --run-name retinanet_base_fold0_1280_e12_b2_s42_v1 --epochs 12 --batch 2 --lr 0.00125 --eval-every 2 --test
python tools/queue/reeval_last.py outputs/yolo26s_base_fold0_1280_e25_s42_v1 --model yolo --imgsz 1280
python -m src.anchors_kmeans
python -m src.tide outputs/frcnn_base_fold0_1280_e12_b2_s42_v1/val_predictions.json --out experiments/tide_frcnn
python experiments/make_summary.py
# Final models (all training clips; scored on the official val/ split) and the ensemble
python -m src.tv_train --arch fasterrcnn --run-name frcnn_final_all_step2_lowres_e6_b2_s42_v1 --fold -1 --step 2 --lowres-p 0.3 --epochs 6 --batch 2 --eval-every 2 --test
python -m src.yolo_final --run-name yolo26s_final_all_step2_e12_b3_s42_v1 --epochs 12 --batch 3
python -m src.tv_predict frcnn_final_all_step2_lowres_e6_b2_s42_v1 --tag valall_hflip --hflip --gt prep/coco/val.json --image-root val   # + plain, test, best.pt
python -m src.yolo_final --run-name yolo26s_final_all_step2_e12_b3_s42_v1 --hflip-only --weights best.pt                             # + epoch2.pt
python -m src.ensemble --gt prep/coco/val.json --iou 0.6 --val <8 val files> --test <8 test files> --out outputs/ens8_frcnn_yolo_2ckpt_tta_iou0.6
python -m src.temporal --val outputs/ens8_frcnn_yolo_2ckpt_tta_iou0.6/val_predictions.json --test outputs/ens8_frcnn_yolo_2ckpt_tta_iou0.6/test_predictions.json --k 2 --alpha 0.15 --thr 0.5 --out outputs/ens8_boxboost
```
