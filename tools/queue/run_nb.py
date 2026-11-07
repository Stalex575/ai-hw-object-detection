"""Execute a notebook headless with cwd=REPO/notebooks (as notebooks/04 expects) and always save the outputs.

    RUN_NAME=<run> RUN_CONFIG=configs/<cfg>.yaml python tools/queue/run_nb.py notebooks/04_anchor_free_yolo.ipynb

The kernel inherits this process's environment (RUN_NAME, RUN_CONFIG, ...). The executed copy goes to
outputs/notebooks/<RUN_NAME or notebook stem>.executed.ipynb, never next to the source notebook.
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    REPO / "outputs/notebooks" / f"{os.environ.get('RUN_NAME', src.stem)}.executed.ipynb"
dst.parent.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": str(REPO / "notebooks")}})
try:
    client.execute()
finally:
    nbformat.write(nb, dst)
    print("saved", dst, flush=True)
