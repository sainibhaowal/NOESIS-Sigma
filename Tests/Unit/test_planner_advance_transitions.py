from Core.Planner.plan_models import Goal, Step, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_planner_advance_start_done():
    steps = [Step(step_id="s0", index=0, title="a"), Step(step_id="s1", index=1, title="b")]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])

    planner = PlannerService(InMemoryPlanStore())
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="BALANCED", goals=[goal])

    plan = planner.advance_start_next(plan_id=plan.plan_id)
    assert plan.goals[0].subtasks[0].steps[0].status == "RUNNING"

    plan = planner.advance_mark_done(plan_id=plan.plan_id, outputs={"k": "v"})
    assert plan.goals[0].subtasks[0].steps[0].status == "DONE"

    plan = planner.advance_start_next(plan_id=plan.plan_id)
    assert plan.goals[0].subtasks[0].steps[1].status == "RUNNING"
