# GPU job queue (laptop)

One GPU job at a time (8 GB VRAM, ~10 GB WSL RAM: a second torch process next to a training run can crash WSL).

- `run_queue.sh` runs `outputs/queue/jobs/*.sh` (git-ignored) in name order. State per job in
  `outputs/queue/state/<job>.{running,done,failed,log}`, START/END lines in `outputs/queue/queue.log`.
  A failed job does not stop the queue. With `outputs/queue/HOLD` present it waits for new jobs instead of exiting.
- Add a job: drop `outputs/queue/jobs/NN_name.sh`; the name decides the order (`51b_` runs after `51_`, before `52_`).
- Kill a job: kill the job's bash script first, then its python; delete `state/<job>.{running,failed}` and the partial
  `outputs/<run>/` (or let the job resume from `last.pt` — torchvision jobs retry once with `--resume`).

Launch:

```bash
touch outputs/queue/HOLD
PATH=$PWD/.venv/bin:$PATH setsid nohup tools/queue/run_queue.sh >> outputs/queue/nohup.out 2>&1 < /dev/null & disown
```

Job templates:

```bash
# torchvision (src/tv_train.py); one changed flag against the family baseline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS="--run-name frcnn_giou_fold0_1280_e12_b2_s42_v1 --epochs 12 --batch 2 --eval-every 2 --test --arch fasterrcnn --box-loss giou"
.venv/bin/python -m src.tv_train $ARGS || .venv/bin/python -m src.tv_train $ARGS --resume

# Ultralytics notebook 04 (after O1: configured through the environment)
RUN_NAME=yolo26s_lr2e-4_fold0_1280_e25_s42_v1 RUN_CONFIG=configs/yolo26s_lr2e-4_1280.yaml \
  .venv/bin/python tools/queue/run_nb.py notebooks/04_anchor_free_yolo.ipynb
```

Other tools:

| File | What |
|---|---|
| `vram_probe.py [fasterrcnn\|retinanet]` | largest batch (4 or 2) that fits at the largest jitter size; caps VRAM so overflow raises OOM |
| `reeval_last.py <run> --model yolo\|rtdetr --imgsz N` | re-score an Ultralytics run on `last.pt` into `<run>/last/` |
| `make_share.py <runs...>` | pack weights + prediction JSONs into `outputs/share/` for Drive |
| `make_nb.py` | legacy: string-substituted copy of notebook 04 (before O1) |
| `src/tv_predict.py` | re-inference of a torchvision run (all frames, NMS / RPN variants, test set) |
