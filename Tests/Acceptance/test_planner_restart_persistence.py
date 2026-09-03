import os

import pytest

from Core.Planner.plan_models import Goal, Step, Subtask
from Core.Planner.plan_store_sim import SimPlanStore
from Core.Planner.planner_service import PlannerService
from External.Sim.Storage.warm_store import WarmStore


def test_planner_restart_persistence():
    db_url = os.getenv("SIM_DB_URL", "").strip()
    if not db_url:
        pytest.skip("SIM_DB_URL not set; PostgreSQL required")
    if "sqlite" in db_url.lower():
        pytest.skip("SQLite not allowed")

    store = SimPlanStore(warm=WarmStore(db_url=db_url, echo_sql=False))
    planner = PlannerService(store)

    steps = [Step(step_id="s0", index=0, title="a")]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])

    plan = planner.create_plan(
        tenant_id="t",
        user_id="u",
        session_id="s",
        policy_mode="BALANCED",
        goals=[goal],
    )

    # simulate restart
    store2 = SimPlanStore(warm=WarmStore(db_url=db_url, echo_sql=False))
    plan2 = store2.get(plan.plan_id)
    assert plan2 is not None
    assert plan2.plan_root_hash == plan.plan_root_hash
