from Core.Planner.plan_models import Goal, RollbackHook, RollbackPolicy, Step, Subtask
from Core.Planner.plan_store_memory import InMemoryPlanStore
from Core.Planner.planner_service import PlannerService


def test_rollback_hook_order_reverse():
    steps = [
        Step(step_id="s0", index=0, title="a", rollback_policy=RollbackPolicy.FULL, rollback_hook=RollbackHook(target_ref="A")),
        Step(step_id="s1", index=1, title="b", rollback_policy=RollbackPolicy.FULL, rollback_hook=RollbackHook(target_ref="B")),
    ]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])

    store = InMemoryPlanStore()
    planner = PlannerService(store)
    plan = planner.create_plan(tenant_id="t", user_id="u", session_id="s", policy_mode="BALANCED", goals=[goal])
    v1 = plan.version

    plan = planner.advance_start_next(plan_id=plan.plan_id)
    plan = planner.advance_mark_done(plan_id=plan.plan_id)
    plan = planner.advance_start_next(plan_id=plan.plan_id)
    plan = planner.advance_mark_done(plan_id=plan.plan_id)

    calls: list[str] = []

    def hook_exec(hook):
        calls.append(hook.target_ref)

    planner.rollback_plan(plan_id=plan.plan_id, to_version=v1, mode=RollbackPolicy.FULL, hook_executor=hook_exec)
    assert calls == ["B", "A"]
