"""
Tests for NOESIS Cognitive Dataset Specification implementation.

Tests cover:
- 8-stage compiler pipeline (Spec Section 9)
- Safety screening (Spec Section 11)
- Dataset type routing (Spec Section 7)
- Advanced exports (Spec Section 13)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from Runtime.Models.brain_curriculum.compiler import (
    BrainCurriculumCompiler,
    CompilerStage,
    StageResult,
)
from Runtime.Models.brain_curriculum.safety_screening import SafetyScreener
from Runtime.Models.brain_curriculum.dataset_type_router import DatasetTypeRouter
from Runtime.Models.brain_curriculum.advanced_exporter import AdvancedExporter, DatasetShard
from Runtime.Models.brain_curriculum.schema import SkillEpisode


class TestSafetyScreening:
    """Test Spec Section 11: Data Safety Rules."""

    def test_secret_detection_api_key(self):
        """Detect API keys in episode text."""
        screener = SafetyScreener()
        episode = {
            "request_text": "Use this API_KEY = sk-abc123def456ghi789jkl",
            "response_text": "OK",
        }
        is_safe, violations = screener.screen_episode(episode)
        assert not is_safe
        assert any(v.violation_type == "secret" for v in violations)

    def test_pii_detection_email(self):
        """Detect emails in episode."""
        screener = SafetyScreener()
        episode = {
            "request_text": "Contact me at john.doe@example.com",
            "response_text": "OK",
        }
        is_safe, violations = screener.screen_episode(episode)
        assert not is_safe
        assert any(v.violation_type == "pii" for v in violations)

    def test_clean_episode_passes(self):
        """Clean episodes pass safety screening."""
        screener = SafetyScreener()
        episode = {
            "request_text": "Write a Python function to sort numbers",
            "response_text": "def sort_list(items): return sorted(items)",
        }
        is_safe, violations = screener.screen_episode(episode)
        assert is_safe
        assert len(violations) == 0


class TestDatasetTypeRouting:
    """Test Spec Section 7: Dataset Type Classification (A-E)."""

    def test_type_a_language_classification(self):
        """Type A: Language skill traces."""
        router = DatasetTypeRouter()
        episode = {
            "domain": "language.text",
            "task_type": "writing",
            "request_text": "Write a paragraph about climate change",
            "response_text": "Climate change...",
        }
        classification = router.classify(episode)
        assert classification.dataset_type == "A"
        assert classification.confidence > 0.5

    def test_type_b_code_classification(self):
        """Type B: Code skill traces."""
        router = DatasetTypeRouter()
        episode = {
            "domain": "software.python",
            "task_type": "coding",
            "request_text": "Write a function to calculate fibonacci",
            "response_text": "def fib(n): ...",
        }
        classification = router.classify(episode)
        assert classification.dataset_type == "B"
        assert classification.confidence > 0.5

    def test_type_c_reasoning_classification(self):
        """Type C: Reasoning traces."""
        router = DatasetTypeRouter()
        episode = {
            "domain": "reasoning",
            "task_type": "problem_solving",
            "request_text": "Solve this differential equation",
            "response_text": "The solution is...",
        }
        classification = router.classify(episode)
        # Should detect reasoning characteristics
        assert classification.dataset_type in ["C", "B"]

    def test_type_d_tooluse_classification(self):
        """Type D: Tool-use traces."""
        router = DatasetTypeRouter()
        episode = {
            "domain": "tool.use",
            "task_type": "command_execution",
            "request_text": "List files in /tmp",
            "response_text": "file1.txt file2.txt",
        }
        classification = router.classify(episode)
        # Tool-use domain should trigger type D
        assert classification.dataset_type in ["D", "B"]

    def test_type_e_failure_repair_classification(self):
        """Type E: Failure-repair traces."""
        router = DatasetTypeRouter()
        episode = {
            "domain": "repair",
            "task_type": "error_fixing",
            "request_text": "Fix this broken code",
            "response_text": "The issue was...",
            "failure_corrections": [{"error": "bug", "fix": "corrected"}],
        }
        classification = router.classify(episode)
        # Repair/error domain should indicate type E
        assert classification.dataset_type in ["E", "B"]


class TestAdvancedExporter:
    """Test Spec Section 13: Export Formats."""

    @pytest.fixture
    def temp_export_dir(self, tmp_path):
        """Create temporary export directory."""
        return tmp_path / "exports"

    def test_export_shards(self, temp_export_dir):
        """Export episodes into train/val/test shards."""
        exporter = AdvancedExporter(temp_export_dir)
        
        episodes = [
            {"episode_id": f"ep{i}", "domain": "test", "promotion_label": "promote"}
            for i in range(100)
        ]

        shards = exporter.export_shards(episodes, train_ratio=0.7, val_ratio=0.15)
        
        assert len(shards) > 0
        assert all(isinstance(s, DatasetShard) for s in shards)
        assert sum(s.episode_count for s in shards) == 100

    def test_export_splits(self, temp_export_dir):
        """Export train/val/test split files."""
        exporter = AdvancedExporter(temp_export_dir)
        
        episodes = [
            {"episode_id": f"ep{i}", "domain": "test"}
            for i in range(50)
        ]

        results = exporter.export_splits(episodes)
        
        assert "train" in results
        assert "val" in results
        assert "test" in results
        assert all(Path(p).exists() for p in results.values())

    def test_validate_schema(self, temp_export_dir):
        """Validate episodes against canonical schema."""
        exporter = AdvancedExporter(temp_export_dir)
        
        valid_episode = {
            "episode_id": "ep1",
            "trace_id": "tr1",
            "tenant_id": "tenant1",
            "domain": "test",
            "task_type": "test",
            "request_text": "test",
            "concepts": [],
            "state_trace": [],
            "artifacts": [],
            "verification": {},
            "witness": {},
            "promotion_label": "promote",
        }

        report = exporter.validate_schema([valid_episode])
        assert report.valid_episodes == 1
        assert report.invalid_episodes == 0

    def test_promotion_manifest(self, temp_export_dir):
        """Create promotion manifest."""
        exporter = AdvancedExporter(temp_export_dir)
        
        episodes = [
            {"episode_id": f"ep{i}", "promotion_label": "promote"}
            for i in range(10)
        ] + [
            {"episode_id": f"ep{i}", "promotion_label": "repair"}
            for i in range(10, 20)
        ]

        manifest = exporter.create_promotion_manifest(episodes)
        
        assert manifest.total_candidates == 10
        assert len(manifest.promotion_candidates) == 10

    def test_repair_candidates(self, temp_export_dir):
        """Export repair candidates."""
        exporter = AdvancedExporter(temp_export_dir)
        
        episodes = [
            {"episode_id": f"ep{i}", "promotion_label": "repair"}
            for i in range(5)
        ] + [
            {"episode_id": f"ep{i}", "promotion_label": "promote"}
            for i in range(5, 10)
        ]

        result = exporter.export_repair_candidates(episodes)
        
        assert result["total_candidates"] == 5
        assert len(result["repair_candidates"]) == 5


class TestCompilerPipeline:
    """Test Spec Section 9: 8-Stage Compiler Pipeline."""

    def test_stage_1_admission(self):
        """Stage 1: Source Admission."""
        compiler = BrainCurriculumCompiler()
        
        mock_traces = [
            MagicMock(trace_id="tr1", request_text="test1", tenant_id="tenant1"),
            MagicMock(trace_id=None, request_text="test2", tenant_id="tenant1"),  # Missing trace_id
        ]

        admitted = compiler._stage_admission(mock_traces, tenant_id="tenant1")
        
        assert len(admitted) == 1
        assert admitted[0].trace_id == "tr1"

    def test_stage_2_screening(self):
        """Stage 2: Safety Screening."""
        compiler = BrainCurriculumCompiler()
        
        # Create mock trace with safe content
        safe_trace = MagicMock()
        safe_trace.__dataclass_fields__ = {}
        safe_trace.__dict__ = {
            "request_text": "Write a Python function",
            "response_text": "def func(): pass",
        }

        screened = compiler._stage_screening([safe_trace])
        
        assert len(screened) > 0

    def test_stage_3_normalization(self):
        """Stage 3: Trace Normalization."""
        compiler = BrainCurriculumCompiler()
        
        trace = MagicMock(session_id=None, created_at=None)
        
        normalized = compiler._stage_normalization([trace])
        
        assert len(normalized) == 1
        assert normalized[0].session_id is not None
        assert normalized[0].created_at is not None

    def test_stage_7_routing(self):
        """Stage 7: Memory-target Routing."""
        compiler = BrainCurriculumCompiler()
        
        episode = SkillEpisode(
            episode_id="ep1",
            trace_id="tr1",
            tenant_id="tenant1",
            domain="software.python",
            task_type="coding",
            request_text="Write Python code",
            response_text="def func(): pass",
            concepts=[],
            state_trace=[],
            artifacts=[],
            plans=[],
            verification={},
            witness={},
            promotion_label="promote",
            source_refs=[],
            graph_dict={},
            trajectory_norms=[],
            verifier_score=0.95,
        )

        routed = compiler._stage_routing([episode])
        
        assert len(routed) == 1
        assert hasattr(routed[0], "dataset_type")
        assert routed[0].dataset_type in ["A", "B", "C", "D", "E"]

    def test_stage_results_tracking(self):
        """Verify stage results are tracked."""
        compiler = BrainCurriculumCompiler()
        
        mock_traces = [MagicMock(trace_id="tr1", request_text="test", tenant_id="tenant1")]
        
        compiler._stage_admission(mock_traces, tenant_id="tenant1")
        
        results = compiler.get_stage_results()
        assert len(results) >= 1
        assert results[0].stage == CompilerStage.ADMISSION
        assert results[0].success


class TestIntegrationPipeline:
    """Test complete 8-stage pipeline integration."""

    def test_full_pipeline_with_safe_episode(self, tmp_path):
        """Run full pipeline with safe episode."""
        compiler = BrainCurriculumCompiler()
        
        # Create mock traces
        trace = MagicMock()
        trace.trace_id = "tr1"
        trace.request_text = "Write a sorting function"
        trace.tenant_id = "tenant1"
        trace.session_id = None
        trace.created_at = None
        trace.graph_dict = {"nodes": []}
        trace.response_text = "def sort(x): return sorted(x)"
        trace.__dataclass_fields__ = {}
        trace.__dict__ = {
            "trace_id": "tr1",
            "request_text": "Write a sorting function",
            "tenant_id": "tenant1",
            "response_text": "def sort(x): return sorted(x)",
        }

        # Run pipeline
        episodes = compiler.pipeline(
            [trace],
            tenant_id="tenant1",
            safety_check=True,
            export_format="jsonl",
            output_dir=tmp_path / "output"
        )

        # Verify results
        assert len(episodes) > 0
        
        # Verify all stages were executed
        stage_results = compiler.get_stage_results()
        assert len(stage_results) >= 3  # At least admission, screening, normalization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
