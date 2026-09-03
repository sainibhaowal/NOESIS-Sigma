import os
from pathlib import Path

import pytest

from Core.LongTask.runner import LongTaskRunner
from Core.LongTask.task_store_sim import SimTaskStore
from Core.Planner.plan_models import Goal, Step, Subtask
from Core.Planner.plan_store_sim import SimPlanStore
from Core.Planner.planner_service import PlannerService
from External.Sim.Storage.warm_store import WarmStore


def test_long_task_forced_restart_resume(tmp_path: Path):
    db_url = os.getenv("SIM_DB_URL", "").strip()
    if not db_url:
        pytest.skip("SIM_DB_URL not set; PostgreSQL required")
    if "sqlite" in db_url.lower():
        pytest.skip("SQLite not allowed")

    warm = WarmStore(db_url=db_url, echo_sql=False)
    task_store = SimTaskStore(warm=warm)
    plan_store = SimPlanStore(warm=warm)
    planner = PlannerService(plan_store)

    # plan with 3 steps (requires 3 chunks)
    steps = [Step(step_id=f"s{i}", index=i, title=f"step{i}") for i in range(3)]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="BALANCED", goals=[goal])

    runner = LongTaskRunner(task_store=task_store, planner=planner, artifacts_root=tmp_path / "art", tasks_root=tmp_path / "tasks")
    task = runner.start_task(tenant_id="t", user_id="u", thread_id="th", plan_id=plan.plan_id, policy_mode="BALANCED")

    # run first two chunks
    task = runner.run_chunk(task.task_id)
    task = runner.run_chunk(task.task_id)

    # simulate restart by creating new runner/store
    runner2 = LongTaskRunner(task_store=SimTaskStore(warm=WarmStore(db_url=db_url, echo_sql=False)), planner=planner, artifacts_root=tmp_path / "art", tasks_root=tmp_path / "tasks")
    task = runner2.run_chunk(task.task_id)
    assert task.current_chunk_id >= 2
