"""Usage: vram_probe.py [fasterrcnn|retinanet]. Pick the largest batch (4 or 2) whose peak VRAM at the LARGEST jitter size stays well inside 8 GB.
Caps the allocator so an overflow raises OOM instead of silently spilling into shared system memory."""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from src import tv_train as T
from pathlib import Path
P = Path(__file__).resolve().parents[2] / 'prep'
ARCH = sys.argv[1] if len(sys.argv) > 1 else 'fasterrcnn'
torch.cuda.set_per_process_memory_fraction(0.82)  # ~6.5 GiB of 7.9
ds = T.TrainSet(P / 'yolo/train_step5_masked_fold0_train.txt', P / 'coco/train_step5_fold0_train.json', hflip=0)
ds.items.sort(key=lambda it: -len(it[2]))  # densest frames first
for batch in (4, 2):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        import os; s = int(os.environ.get('PROBE_SIZE', 896))
        model = T.build_model(ARCH, [s], 720, int(s * 16 / 9) + 8, 1280).cuda(); model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scaler = torch.amp.GradScaler('cuda')
        for i in range(6):
            imgs, tgts = T.collate([ds[(i * batch + k) % 64] for k in range(batch)])
            with torch.autocast('cuda', dtype=torch.float16):
                loss = sum(model([x.cuda() for x in imgs], [{k: v.cuda() for k, v in t.items()} for t in tgts]).values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        peak = torch.cuda.max_memory_reserved() / 2**30
        print(f'batch {batch}: peak {peak:.2f} GiB', file=sys.stderr)
        print(batch); break
    except torch.OutOfMemoryError:
        print(f'batch {batch}: OOM', file=sys.stderr)
        del model, opt
else:
    print(1)
