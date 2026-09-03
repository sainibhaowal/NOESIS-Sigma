# Tests/Skills/test_skills.py
#
# Sprint B3 — Skill System
#
# Tests for:
#   Skill / SkillStep / SkillChain data models
#   Built-in operator libraries (code_ops, plan_ops)
#   TaskPlanner: decompose(), registry, cosine retrieval, fallback
#   SkillExecutor: run(), StepResult, ExecutionResult
#   FeedbackStore: record(), EMA success_rate update
#   ContextBundle skill_procedures field + build_context_tensor
#   GraphExtractor: PLAN nodes from skill_procedures
#   ThoughtGraph.summary() skill_plans count
#   OscOrchestrator._init_task_planner() graceful init

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. Skill model
# ---------------------------------------------------------------------------

def test_skill_step_to_dict_roundtrip():
    from External.Skills.Models.skill import SkillStep
    step = SkillStep(
        step_id="s1",
        operator="write_function",
        inputs=["signature", "tests"],
        outputs=["code"],
        description="Write a function.",
    )
    d = step.to_dict()
    restored = SkillStep.from_dict(d)
    assert restored.operator == "write_function"
    assert restored.inputs == ["signature", "tests"]
    assert restored.outputs == ["code"]


def test_skill_to_dict_roundtrip():
    from External.Skills.Models.skill import Skill, SkillStep
    step = SkillStep(step_id="s1", operator="op", inputs=["x"], outputs=["y"], description="d")
    skill = Skill(
        skill_id="sk1", name="Test skill", domain="coding",
        description="Does a thing.", steps=[step], source_ref="test",
    )
    d = skill.to_dict()
    restored = Skill.from_dict(d)
    assert restored.skill_id == "sk1"
    assert restored.domain == "coding"
    assert len(restored.steps) == 1
    assert restored.steps[0].operator == "op"


def test_skill_chain_plan_node_descriptions():
    from External.Skills.Models.skill import Skill, SkillChain, SkillStep
    step = SkillStep(step_id="s1", operator="write_function", inputs=[], outputs=[], description="")
    skill = Skill(skill_id="sk1", name="My skill", domain="coding",
                  description="does stuff", steps=[step])
    chain = SkillChain(chain_id="c1", goal="write code", skills=[skill], estimated_steps=1)
    descs = chain.plan_node_descriptions()
    assert len(descs) == 1
    desc, sid = descs[0]
    assert "coding" in desc
    assert "My skill" in desc
    assert sid == "sk1"


def test_skill_chain_is_direct_false_for_real_skill():
    from External.Skills.Models.skill import Skill, SkillChain
    skill = Skill(skill_id="sk1", name="Real", domain="coding", description="real skill")
    chain = SkillChain(chain_id="c1", goal="g", skills=[skill])
    assert not chain.is_direct()


def test_skill_chain_is_direct_true_for_fallback():
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner(skills=[])
    chain = planner.decompose(goal="", mode="chat")
    assert chain.is_direct()


# ---------------------------------------------------------------------------
# 2. Built-in operator libraries
# ---------------------------------------------------------------------------

def test_builtin_code_skills_non_empty():
    from External.Skills.Operators.code_ops import BUILTIN_CODE_SKILLS
    assert len(BUILTIN_CODE_SKILLS) >= 3
    domains = {sk.domain for sk in BUILTIN_CODE_SKILLS}
    assert "coding" in domains


def test_builtin_plan_skills_non_empty():
    from External.Skills.Operators.plan_ops import BUILTIN_PLAN_SKILLS
    assert len(BUILTIN_PLAN_SKILLS) >= 3
    domains = {sk.domain for sk in BUILTIN_PLAN_SKILLS}
    assert "planning" in domains


def test_builtin_skills_have_descriptions():
    from External.Skills.Operators import BUILTIN_SKILLS
    for sk in BUILTIN_SKILLS:
        assert sk.description, f"Skill '{sk.name}' has no description"
        assert sk.name


def test_builtin_code_operators_have_correct_operators():
    from External.Skills.Operators.code_ops import BUILTIN_CODE_SKILLS
    all_ops = {step.operator for sk in BUILTIN_CODE_SKILLS for step in sk.steps}
    assert "write_function" in all_ops
    assert "run_tests" in all_ops
    assert "fix_error" in all_ops or "refactor" in all_ops


def test_builtin_plan_operators_have_correct_operators():
    from External.Skills.Operators.plan_ops import BUILTIN_PLAN_SKILLS
    all_ops = {step.operator for sk in BUILTIN_PLAN_SKILLS for step in sk.steps}
    assert "decompose_goal" in all_ops
    assert "sequence_steps" in all_ops
    assert "verify_plan" in all_ops


# ---------------------------------------------------------------------------
# 3. TaskPlanner
# ---------------------------------------------------------------------------

def test_task_planner_seeded_with_builtins():
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    assert planner.registry_size() > 0


def test_task_planner_decompose_returns_chain():
    from External.Skills.Models.skill import SkillChain
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("write a function that reverses a list", mode="code")
    assert isinstance(chain, SkillChain)
    assert chain.goal != ""


def test_task_planner_code_goal_returns_coding_skills():
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("write and test a Python function", mode="code")
    if not chain.is_direct():
        domains = {sk.domain for sk in chain.skills}
        assert "coding" in domains


def test_task_planner_plan_goal_returns_planning_skills():
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("decompose this goal into a plan", mode="plan")
    if not chain.is_direct():
        domains = {sk.domain for sk in chain.skills}
        assert "planning" in domains


def test_task_planner_empty_goal_returns_direct():
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("", mode="chat")
    assert chain.is_direct()


def test_task_planner_chat_mode_still_returns_chain():
    from External.Skills.Models.skill import SkillChain
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("hello how are you", mode="chat")
    assert isinstance(chain, SkillChain)


def test_task_planner_register_custom_skill():
    from External.Skills.Models.skill import Skill
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner(skills=[])
    assert planner.registry_size() == 0
    custom = Skill(
        skill_id="custom1", name="Custom skill", domain="tool_use",
        description="Use a custom external tool to fetch data.",
    )
    planner.register(custom)
    assert planner.registry_size() == 1
    assert custom.embedding  # embedding was computed on register


def test_task_planner_high_score_match():
    from External.Skills.Models.skill import Skill
    from External.Skills.Planner.task_planner import TaskPlanner
    # Register a skill with very specific description
    skill = Skill(
        skill_id="exact1", name="Run Python unit tests",
        domain="coding",
        description="Execute Python unit tests using pytest and report failures.",
    )
    planner = TaskPlanner(skills=[skill])
    chain = planner.decompose("run python unit tests with pytest", mode="code")
    assert not chain.is_direct()
    assert chain.skills[0].skill_id == "exact1"


# ---------------------------------------------------------------------------
# 4. SkillExecutor
# ---------------------------------------------------------------------------

def test_skill_executor_run_returns_result():
    from External.Skills.Execution.skill_executor import SkillExecutor
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("write a function", mode="code")
    executor = SkillExecutor()
    result = executor.run(chain)
    assert result.chain_id == chain.chain_id
    assert result.success is True
    assert len(result.steps) > 0


def test_skill_executor_step_results_have_outputs():
    from External.Skills.Execution.skill_executor import SkillExecutor
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("plan a project roadmap", mode="plan")
    executor = SkillExecutor()
    result = executor.run(chain)
    for sr in result.steps:
        assert sr.operator != ""
        assert isinstance(sr.success, bool)


def test_skill_executor_direct_chain_runs():
    from External.Skills.Execution.skill_executor import SkillExecutor
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner(skills=[])
    chain = planner.decompose("", mode="chat")  # direct fallback
    executor = SkillExecutor()
    result = executor.run(chain)
    assert result.success is True
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# 5. FeedbackStore
# ---------------------------------------------------------------------------

def test_feedback_store_records_outcome():
    from External.Skills.Execution.skill_executor import SkillExecutor
    from External.Skills.Feedback.feedback_store import FeedbackStore
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner()
    chain = planner.decompose("write code", mode="code")
    executor = SkillExecutor()
    result = executor.run(chain)
    store = FeedbackStore()
    skill = chain.skills[0] if chain.skills else None
    store.record(result, skill=skill)
    assert store.record_count() == 1


def test_feedback_store_updates_success_rate():
    from External.Skills.Execution.skill_executor import ExecutionResult, SkillExecutor
    from External.Skills.Feedback.feedback_store import FeedbackStore
    from External.Skills.Models.skill import Skill
    skill = Skill(skill_id="fb1", name="Test skill", domain="coding",
                  description="test", success_rate=1.0)
    store = FeedbackStore()
    # Simulate a failed execution by patching the result
    executor = SkillExecutor()
    from External.Skills.Planner.task_planner import TaskPlanner
    planner = TaskPlanner(skills=[skill])
    chain = planner.decompose("test task", mode="code")
    result = executor.run(chain)
    # Manually set failure
    result_fail = ExecutionResult(
        execution_id="e1", chain_id="c1", goal="test",
        steps=result.steps, success=False, elapsed_ms=1.0
    )
    store.record(result_fail, skill=skill)
    assert skill.success_rate < 1.0  # EMA pushed down by failure


def test_feedback_store_ema_is_smooth():
    from External.Skills.Execution.skill_executor import ExecutionResult
    from External.Skills.Feedback.feedback_store import FeedbackStore
    from External.Skills.Models.skill import Skill
    skill = Skill(skill_id="fb2", name="EMA test", domain="coding",
                  description="test", success_rate=1.0)
    store = FeedbackStore()
    # Record 5 failures
    for i in range(5):
        result = ExecutionResult(
            execution_id=f"e{i}", chain_id="c1", goal="g",
            steps=[], success=False, elapsed_ms=1.0
        )
        store.record(result, skill=skill)
    # Rate should have moved toward 0 but not reached it (EMA smoothing)
    assert 0.0 < skill.success_rate < 1.0


# ---------------------------------------------------------------------------
# 6. ContextBundle skill_procedures field
# ---------------------------------------------------------------------------

def test_context_bundle_skill_procedures_field():
    from Core.Cognition.fusion import ContextBundle
    bundle = ContextBundle(
        request_text="write code",
        skill_procedures=[("[coding] Write and test a Python function: write_function → run_tests", "sk1")],
        state_dim=64,
    )
    assert len(bundle.skill_procedures) == 1
    assert bundle.skill_procedures[0][1] == "sk1"


def test_context_bundle_skill_procedures_defaults_empty():
    from Core.Cognition.fusion import ContextBundle
    bundle = ContextBundle(request_text="hello")
    assert bundle.skill_procedures == []


def test_build_context_tensor_with_skill_procedures():
    from Core.Cognition.fusion import ContextBundle, build_context_tensor
    bundle = ContextBundle(
        request_text="write a sorting function",
        skill_procedures=[
            ("[coding] Write and test: write_function → run_tests", "sk1"),
            ("[coding] Fix bug: fix_error → run_tests", "sk2"),
        ],
        state_dim=128,
    )
    tensor = build_context_tensor(bundle)
    assert tensor.shape == torch.Size([128])
    assert abs(tensor.norm().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 7. GraphExtractor PLAN nodes
# ---------------------------------------------------------------------------

def test_graph_extractor_creates_plan_nodes():
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle_args = dict(
        request_text="write a function",
        skill_procedures=[("[coding] My skill: op1 → op2", "skill-id-1")],
        state_dim=64,
    )
    from Core.Cognition.fusion import ContextBundle
    bundle = ContextBundle(**bundle_args)
    graph = extractor.extract(trajectory=[], bundle=bundle)

    plan_nodes = graph.get_nodes_by_type(NodeType.PLAN)
    assert len(plan_nodes) == 1
    assert plan_nodes[0].source_ref == "skill:skill-id-1"
    assert plan_nodes[0].confidence == 0.9


def test_graph_extractor_plan_nodes_have_skill_prefix():
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = ContextBundle(
        request_text="plan a project",
        skill_procedures=[
            ("[planning] Decompose: decompose_goal", "plan-skill-1"),
            ("[planning] Sequence: sequence_steps", "plan-skill-2"),
        ],
        state_dim=64,
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)
    plan_nodes = graph.get_nodes_by_type(NodeType.PLAN)
    refs = {n.source_ref for n in plan_nodes}
    assert "skill:plan-skill-1" in refs
    assert "skill:plan-skill-2" in refs


def test_graph_extractor_plan_nodes_do_not_affect_fact_nodes():
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = ContextBundle(
        request_text="write code",
        sim_memories=[("some memory", "sim-1")],
        skill_procedures=[("[coding] Write: write_function", "sk1")],
        state_dim=64,
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)
    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    plan_nodes = graph.get_nodes_by_type(NodeType.PLAN)
    sim_nodes = [n for n in fact_nodes if (n.source_ref or "").startswith("sim:")]
    assert len(sim_nodes) == 1
    assert len(plan_nodes) == 1


def test_thought_graph_summary_skill_plans_count():
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor

    extractor = GraphExtractor()
    bundle = ContextBundle(
        request_text="plan this",
        skill_procedures=[
            ("[planning] Step 1", "sk1"),
            ("[planning] Step 2", "sk2"),
        ],
        state_dim=64,
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)
    s = graph.summary()
    assert s["skill_plans"] == 2
    assert "node_count" in s
    assert "world_model_facts" in s


def test_thought_graph_summary_skill_plans_zero_when_none():
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor

    extractor = GraphExtractor()
    bundle = ContextBundle(request_text="chat message", state_dim=64)
    graph = extractor.extract(trajectory=[], bundle=bundle)
    s = graph.summary()
    assert s["skill_plans"] == 0


# ---------------------------------------------------------------------------
# 8. OscOrchestrator integration
# ---------------------------------------------------------------------------

def test_osc_orchestrator_has_task_planner():
    from External.Orchestrator.osc_chat import OscOrchestrator
    result = OscOrchestrator._init_task_planner()
    assert result is not None


def test_osc_orchestrator_task_planner_has_skills():
    from External.Orchestrator.osc_chat import OscOrchestrator
    planner = OscOrchestrator._init_task_planner()
    assert planner.registry_size() > 0


# ---------------------------------------------------------------------------
# 9. Full pipeline: planner → bundle → graph
# ---------------------------------------------------------------------------

def test_planner_to_bundle_to_graph_pipeline():
    """Full pipeline: TaskPlanner → skill_procedures → ContextBundle → GraphExtractor → PLAN nodes."""
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType
    from External.Skills.Planner.task_planner import TaskPlanner

    planner = TaskPlanner()
    chain = planner.decompose("write and test a Python sorting function", mode="code")
    skill_procedures = chain.plan_node_descriptions()

    bundle = ContextBundle(
        request_text="write and test a Python sorting function",
        skill_procedures=skill_procedures,
        state_dim=128,
    )
    extractor = GraphExtractor()
    graph = extractor.extract(trajectory=[], bundle=bundle)

    plan_nodes = graph.get_nodes_by_type(NodeType.PLAN)
    # For a code goal we expect at least one skill to be retrieved
    if skill_procedures:
        assert len(plan_nodes) == len(skill_procedures)
        for pn in plan_nodes:
            assert (pn.source_ref or "").startswith("skill:")

    summary = graph.summary()
    assert summary["skill_plans"] == len(plan_nodes)
