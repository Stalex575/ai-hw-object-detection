"""Legacy (before Oleh's O1 env-var notebook): make a run-specific copy of notebooks/04 (Oleh's) with only run name/config/model class changed."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
run_name, config, arch, family = sys.argv[1:5]
nb = json.loads((REPO / 'notebooks/04_anchor_free_yolo.ipynb').read_text())
subs = [("RUN_NAME = 'yolo26s_smoke_fold0_640_s42_v1'", f"RUN_NAME = '{run_name}'"),
        ("configs/yolo26_smoke.yaml", config),
        ("shutil.copy2(OUT / 'submission.csv', WORK / 'submission.csv')",
         "shutil.copy2(OUT / 'submission.csv', OUT.parent / f'{RUN_NAME}_submission.csv')"),
        ("archive = WORK / 'day1_results.zip'", "archive = OUT.parent / f'{RUN_NAME}_results.zip'"),
        ("print('Submission:', WORK / 'submission.csv')", "print('Submission:', OUT.parent / f'{RUN_NAME}_submission.csv')"),
        ("'family': 'YOLO26s'", f"'family': '{family}'")]
if arch == 'rtdetr':
    subs.append(("from ultralytics import YOLO, settings", "from ultralytics import RTDETR as YOLO, settings"))
code = json.dumps(nb)
for old, new in subs:
    o, n = json.dumps(old)[1:-1], json.dumps(new)[1:-1]
    if o not in code:
        sys.exit(f'pattern not found: {old}')
    code = code.replace(o, n)
nb = json.loads(code)
nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
    f'**Generated run copy of `notebooks/04_anchor_free_yolo.ipynb`** — run `{run_name}`, config `{config}`, model class `{arch}`. '
    'Executed headless on the RTX 5060 Laptop (WSL2).']})
out = REPO / 'outputs/notebooks' / f'{run_name}.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(out)
