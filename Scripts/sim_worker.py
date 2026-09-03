from __future__ import annotations

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


import time

from loguru import logger

from External.Sim.Api.sim_api import get_sim_state


def main() -> None:
    if not os.getenv("SIM_DB_URL"):
        print("SKIP: SIM_DB_URL not set")
        return
    get_sim_state.cache_clear()
    state = get_sim_state()
    compactor = state["compactor"]

    logger.info("SIM worker started (polling sim_jobs)...")
    while True:
        jid = compactor.run_one_job()
        if jid is None:
            time.sleep(0.5)
            continue
        # tiny backoff after job
        time.sleep(0.05)


if __name__ == "__main__":
    main()
