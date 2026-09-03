from Core.Planner.plan_models import Goal, RollbackPolicy, Step, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_planner_rollback_after_failure():
    steps = [Step(step_id="s0", index=0, title="a"), Step(step_id="s1", index=1, title="b")]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])

    planner = PlannerService(InMemoryPlanStore())
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="BALANCED", goals=[goal])
    v1 = plan.version

    plan = planner.advance_start_next(plan_id=plan.plan_id)
    plan = planner.advance_mark_fail(plan_id=plan.plan_id)

    rolled = planner.rollback_plan(plan_id=plan.plan_id, to_version=v1, mode=RollbackPolicy.PLAN_ONLY)
    assert rolled.goals[0].subtasks[0].steps[0].status == "PENDING"
    assert rolled.status != "FAIL"
