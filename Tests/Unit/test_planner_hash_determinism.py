from Core.Planner.plan_hash import plan_sha256
from Core.Planner.plan_models import Goal, Plan, Step, Subtask


def test_plan_hash_deterministic_with_reordered_steps():
    steps_a = [
        Step(step_id="s1", index=1, title="b"),
        Step(step_id="s0", index=0, title="a"),
    ]
    steps_b = [
        Step(step_id="s0", index=0, title="a"),
        Step(step_id="s1", index=1, title="b"),
    ]
    sub_a = Subtask(subtask_id="st", title="sub", steps=steps_a)
    sub_b = Subtask(subtask_id="st", title="sub", steps=steps_b)
    goal_a = Goal(goal_id="g", title="goal", subtasks=[sub_a])
    goal_b = Goal(goal_id="g", title="goal", subtasks=[sub_b])
    plan_a = Plan(plan_id="p", goals=[goal_a])
    plan_b = Plan(plan_id="p", goals=[goal_b])
    assert plan_sha256(plan_a) == plan_sha256(plan_b)
