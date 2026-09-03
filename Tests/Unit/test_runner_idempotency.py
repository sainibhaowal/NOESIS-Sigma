from pathlib import Path

from Core.LongTask.runner import LongTaskRunner
from Core.LongTask.task_store_memory import InMemoryTaskStore
from Core.Planner.plan_models import Goal, Step, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_runner_idempotency(tmp_path: Path):
    plan_store = InMemoryPlanStore()
    planner = PlannerService(plan_store)
    goal = Goal(goal_id="g", title="goal", subtasks=[Subtask(subtask_id="st", title="sub", steps=[Step(step_id="s0", index=0, title="a")])])
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="BALANCED", goals=[goal])

    task_store = InMemoryTaskStore()
    runner = LongTaskRunner(task_store=task_store, planner=planner, artifacts_root=tmp_path / "art", tasks_root=tmp_path / "tasks")
    task = runner.start_task(tenant_id="t", user_id="u", thread_id="th", plan_id=plan.plan_id, policy_mode="BALANCED")

    task1 = runner.run_chunk(task.task_id)
    task2 = runner.run_chunk(task.task_id)
    assert task2.current_chunk_id >= task1.current_chunk_id
