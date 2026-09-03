from Core.Planner.invariants import PlanInvariantError, assert_invariants
from Core.Planner.plan_models import Goal, Plan, PlanStatus, Step, StepStatus, Subtask


def _mk_plan() -> Plan:
    steps = [
        Step(step_id="s0", index=0, title="a"),
        Step(step_id="s1", index=1, title="b"),
    ]
    sub = Subtask(subtask_id="st", title="sub", steps=steps)
    goal = Goal(goal_id="g", title="goal", subtasks=[sub])
    return Plan(plan_id="p", goals=[goal])


def test_invariants_ok():
    plan = _mk_plan()
    assert_invariants(plan)


def test_invariants_two_running():
    plan = _mk_plan()
    plan.goals[0].subtasks[0].steps[0].status = StepStatus.RUNNING
    plan.goals[0].subtasks[0].steps[1].status = StepStatus.RUNNING
    try:
        assert_invariants(plan)
        assert False, "should have failed"
    except PlanInvariantError:
        pass


def test_invariants_blocked_plan_status():
    plan = _mk_plan()
    plan.goals[0].subtasks[0].steps[0].status = StepStatus.BLOCKED
    plan.status = PlanStatus.PENDING
    try:
        assert_invariants(plan)
        assert False, "should have failed"
    except PlanInvariantError:
        pass
