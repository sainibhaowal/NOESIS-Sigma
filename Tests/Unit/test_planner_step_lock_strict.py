from Core.Planner.plan_models import Goal, Step, StepKind, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_step_lock_strict_blocks_tool_step():
    steps = [Step(step_id="s0", index=0, title="tool", kind=StepKind.TOOL)]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])

    planner = PlannerService(InMemoryPlanStore())
    plan = planner.create_plan(
        tenant_id="t",
        user_id="u",
        session_id="s",
        policy_mode="STRICT",
        goals=[goal],
    )
    plan = planner.advance_start_next(plan_id=plan.plan_id)
    step = plan.goals[0].subtasks[0].steps[0]
    assert step.lock_state == "LOCKED"
    assert plan.status == "BLOCKED"
