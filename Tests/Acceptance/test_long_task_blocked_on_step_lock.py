from pathlib import Path

from Core.LongTask.runner import LongTaskRunner
from Core.LongTask.task_store_memory import InMemoryTaskStore
from Core.Planner.plan_models import Goal, Step, StepKind, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_long_task_blocked_on_step_lock(tmp_path: Path):
    plan_store = InMemoryPlanStore()
    planner = PlannerService(plan_store)
    steps = [Step(step_id="s0", index=0, title="tool", kind=StepKind.TOOL)]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="STRICT", goals=[goal])

    runner = LongTaskRunner(task_store=InMemoryTaskStore(), planner=planner, artifacts_root=tmp_path / "art", tasks_root=tmp_path / "tasks")
    task = runner.start_task(tenant_id="t", user_id="u", thread_id="th", plan_id=plan.plan_id, policy_mode="STRICT")
    task = runner.run_chunk(task.task_id)
    assert task.state == "BLOCKED"
