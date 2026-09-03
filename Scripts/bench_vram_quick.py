
import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo = Path(__file__).resolve()
    for _ in range(6):
        if (repo / 'pyproject.toml').exists():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

_ensure_repo_root()

# --- repo-root sys.path bootstrap (so "Tools", "Core", etc. resolve) ---
import os
import sys
import warnings

import torch

from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.icnn import ICNNDirectGrad
from Core.OSC.params import load_params

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- end bootstrap ---

from External.Tools.logging_setup import setup_logging

setup_logging()  # file DEBUG, console WARNING

warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

p = load_params()
p.state_dim = 2048
p.icnn = ICNNDirectGrad(d=p.state_dim, m=256, dtype=p.dtype or torch.float32,
                        device=p.device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        ws_dtype=torch.float32)

eng = OperatorSplitEngine(p)
x = torch.randn(p.state_dim, device=eng.device, dtype=eng.dtype)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

for _ in range(2000):
    x = eng.step(x)

if torch.cuda.is_available():
    print("peak MB:", torch.cuda.max_memory_allocated()/1e6)
else:
    print("CPU run")
