import pytest
import os
from fastapi.testclient import TestClient
from app.main import app
from app.services.cost_calculator import CostCalculator
from app.services.experience_buffer import ExperienceBufferService, ExperienceRecord
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest, DecisionResponse, DecisionTrace
from app.schemas.provider import ProviderGenerationResponse, TokenUsage
from app.schemas.response import ResponseGenerationResponse
from app.schemas.verification import VerificationResponse
from app.schemas.reward import RewardComputeResponse, RewardBreakdown, ComponentContribution
from app.schemas.orchestration import OrchestrationResponse
from app.schemas.complex import ComplexTaskPlan, SubTask

client = TestClient(app)

def test_rl_production_override_invariant():
    """Requirement 1: RLContextualBanditPolicy is production routing authority; safe fallback is BaselineAdaptivePolicy."""
    engine = AdaptiveDecisionEngine()
    
    # Local request
    req_local = DecisionRequest(text="Explain binary search tree", execution_mode="local")
    dec_local = engine.decide(req_local)
    assert dec_local.policy == "rl_contextual_bandit_policy"
    assert dec_local.shadow_rl_decision is not None
    assert dec_local.shadow_rl_decision.get("production_policy") == "rl_contextual_bandit_policy"
    assert dec_local.shadow_rl_decision.get("fallback_used") is False

    # Online request
    req_online = DecisionRequest(text="Write a complex distributed lock", execution_mode="online")
    dec_online = engine.decide(req_online)
    assert dec_online.policy == "rl_contextual_bandit_policy"
    assert dec_online.shadow_rl_decision is not None
    assert dec_online.shadow_rl_decision.get("production_policy") == "rl_contextual_bandit_policy"
    assert dec_online.shadow_rl_decision.get("fallback_used") is False

def test_cost_calculator_zero_local():
    """Requirement 2: Local Ollama execution models have cost = 0.0 and cost_source = 'zero_local'."""
    calc = CostCalculator()
    
    local_models = ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]
    for m in local_models:
        res = calc.calculate_cost(provider="ollama", model_id=m, input_tokens=1000, output_tokens=500, execution_mode="local")
        assert res["cost"] == 0.0
        assert res["cost_source"] == "zero_local"
        assert res["cost_currency"] == "USD"

def test_cost_calculator_cloud_pricing():
    """Requirement 3: Cloud models calculate cost accurately based on token pricing table."""
    calc = CostCalculator()
    
    # Gemini 3.5 Flash: $0.00015/1k input ($0.15/1M), $0.00060/1k output ($0.60/1M)
    res_g = calc.calculate_cost(provider="gemini", model_id="gemini-3.5-flash", input_tokens=1_000_000, output_tokens=1_000_000, execution_mode="online")
    assert res_g["cost"] == 0.75
    assert res_g["cost_source"] == "configured_pricing_estimate"
    assert res_g["cost_currency"] == "USD"

    # Mistral Small: $0.00020/1k input, $0.00060/1k output
    res_m = calc.calculate_cost(provider="mistral", model_id="mistral-small-latest", input_tokens=1_000_000, output_tokens=1_000_000, execution_mode="online")
    assert res_m["cost"] == 0.80
    assert res_m["cost_source"] == "configured_pricing_estimate"

    # Groq Llama 3 70B: $0.00059/1k input, $0.00079/1k output
    res_gr = calc.calculate_cost(provider="groq", model_id="llama-3.3-70b-versatile", input_tokens=100_000, output_tokens=100_000, execution_mode="online")
    assert res_gr["cost"] > 0.0
    assert res_gr["cost_source"] == "configured_pricing_estimate"

def test_provider_response_cost_fields():
    """Requirement 4: ProviderGenerationResponse correctly holds cost, cost_currency, and cost_source metadata."""
    resp = ProviderGenerationResponse(
        provider="gemini",
        model_id="gemini-2.0-flash",
        generated_text="Test output",
        execution_status="success",
        cost=0.0015,
        cost_currency="USD",
        cost_source="configured_pricing_estimate",
        usage=TokenUsage(input_tokens=1000, output_tokens=250, total_tokens=1250),
        latency_ms=120.0,
        success=True
    )
    assert resp.cost == 0.0015
    assert resp.cost_currency == "USD"
    assert resp.cost_source == "configured_pricing_estimate"

def test_synthesis_zero_cost_assembly():
    """Requirement 5: S2 deterministic assembly synthesis cost is 0.0."""
    subtasks = [
        SubTask(
            task_id="TASK-1",
            description="Analyze problem",
            category="general",
            assigned_model="qwen-coder-3b",
            provider="ollama",
            execution_success=True,
            cost=0.0,
            cost_source="zero_local"
        )
    ]
    
    plan = ComplexTaskPlan(
        is_complex=True,
        original_prompt="Test complex workflow",
        subtasks=subtasks,
        execution_levels=[["TASK-1"]],
        total_subtasks=1,
        plan_latency_ms=50.0,
        aggregated_response="Combined results",
        execution_success=True,
        total_execution_latency_ms=1250.0,
        synthesis_cost=0.0,
        total_workflow_cost=0.0
    )
    
    assert plan.synthesis_cost == 0.0
    assert plan.total_workflow_cost == 0.0

def test_experience_buffer_cost_recording_and_aggregation(tmp_path):
    """Requirement 6: ExperienceBuffer records costs and get_cost_metrics aggregates them accurately."""
    buf = ExperienceBufferService(persistence_path=str(tmp_path / "cost_buffer.jsonl"))
    buf.clear()
    
    dummy_trace = DecisionTrace(
        decision_id="dec_1",
        intent="CODE_GEN",
        is_ambiguous=False,
        complexity_level="SIMPLE",
        complexity_score=0.2,
        resource_summary={"vram_used_gb": 4.0},
        selected_model="qwen-coder-3b",
        decision_score=0.8,
        policy="baseline_adaptive_policy",
        execution_mode="local",
        decision_latency_ms=5.0
    )
    
    dummy_decision = DecisionResponse(
        text="Write a script",
        selected_model="qwen-coder-3b",
        decision_score=0.8,
        policy="baseline_adaptive_policy",
        reasoning=["best fit"],
        candidates=[],
        decision_trace=dummy_trace,
        shadow_rl_decision={"selected_model": "qwen-coder-3b", "production_override": False},
        intent_info={"primary_intent": "CODE_GEN"},
        complexity_info={"assessed_complexity": "SIMPLE"},
        total_pipeline_latency_ms=10.0
    )

    dummy_breakdown = RewardBreakdown(
        quality=ComponentContribution(score=0.9, weight=0.3, contribution=0.27),
        completeness=ComponentContribution(score=0.9, weight=0.2, contribution=0.18),
        relevance=ComponentContribution(score=0.9, weight=0.25, contribution=0.225),
        verification=ComponentContribution(score=1.0, weight=0.15, contribution=0.15),
        execution=ComponentContribution(score=1.0, weight=0.10, contribution=0.10)
    )
    
    # Record a local experience
    orchestration_local = OrchestrationResponse(
        run_id="run_loc_1",
        success=True,
        prompt="Write a Python script",
        selected_model="qwen-coder-3b",
        decision_score=0.8,
        decision=dummy_decision,
        generation=ResponseGenerationResponse(
            provider="ollama",
            model_id="qwen-coder-3b",
            generated_text="print('hello')",
            execution_status="completed",
            cost=0.0,
            cost_source="zero_local",
            usage=TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            latency_ms=150.0,
            success=True
        ),
        verification=VerificationResponse(
            verified=True,
            prompt="Write a Python script",
            selected_model="qwen-coder-3b",
            response_present=True,
            relevance_score=0.9,
            completeness_score=0.9,
            structural_quality_score=0.9,
            verification_reasoning=["passed"],
            verified_model="qwen-coder-3b",
            verification_status="PASSED",
            quality_score=0.95,
            verification_latency_ms=10.0,
            checks_passed=["syntax"]
        ),
        reward=RewardComputeResponse(
            success=True,
            selected_model="qwen-coder-3b",
            reward=0.95,
            reward_breakdown=dummy_breakdown,
            reward_status="COMPUTED",
            reasoning=["passed"],
            latency_ms=5.0
        ),
        pipeline_latency_ms=150.0
    )
    
    buf.record_from_orchestration(orchestration_local)
    
    # Record a cloud experience
    orchestration_cloud = OrchestrationResponse(
        run_id="run_cld_1",
        success=True,
        prompt="Write an enterprise microservice architecture",
        selected_model="gemini-2.0-flash",
        decision_score=0.9,
        decision=dummy_decision,
        generation=ResponseGenerationResponse(
            provider="gemini",
            model_id="gemini-2.0-flash",
            generated_text="Architecture spec...",
            execution_status="completed",
            cost=0.0025,
            cost_source="configured_pricing_estimate",
            usage=TokenUsage(input_tokens=5000, output_tokens=3000, total_tokens=8000),
            latency_ms=450.0,
            success=True
        ),
        verification=VerificationResponse(
            verified=True,
            prompt="Write an enterprise microservice architecture",
            selected_model="gemini-2.0-flash",
            response_present=True,
            relevance_score=0.9,
            completeness_score=0.9,
            structural_quality_score=0.9,
            verification_reasoning=["passed"],
            verified_model="gemini-2.0-flash",
            verification_status="PASSED",
            quality_score=0.9,
            verification_latency_ms=15.0,
            checks_passed=["relevance"]
        ),
        reward=RewardComputeResponse(
            success=True,
            selected_model="gemini-2.0-flash",
            reward=0.88,
            reward_breakdown=dummy_breakdown,
            reward_status="COMPUTED",
            reasoning=["passed"],
            latency_ms=5.0
        ),
        pipeline_latency_ms=450.0
    )
    
    buf.record_from_orchestration(orchestration_cloud)
    
    metrics = buf.get_cost_metrics()
    
    assert metrics["total_api_cost"] == 0.0025
    assert metrics["cloud_requests_count"] == 1
    assert metrics["local_requests_count"] == 1
    assert metrics["total_tokens"] == 8070
    assert "gemini" in metrics["cost_by_provider"]
    assert "ollama" in metrics["cost_by_provider"]
    assert metrics["cost_by_provider"]["gemini"]["total_cost"] == 0.0025
    assert metrics["cost_by_provider"]["ollama"]["total_cost"] == 0.0

def test_complex_workflow_cost_aggregation_includes_failover_and_ignores_s2_synthesis_cost():
    """Complex execution totals must include all billable provider calls; default S2 synthesis remains zero-cost."""

    class DummyDynamicDecomposer:
        def is_decomposable_complex(self, *args, **kwargs):
            return True

        def decompose(self, *args, **kwargs):
            return [
                SubTask(
                    task_id="TASK-1",
                    description="Primary model subtask",
                    category="general",
                    assigned_model="gemini-3.5-flash",
                    provider="Google Gemini API",
                    execution_success=True,
                    cost=0.0025,
                    cost_source="configured_pricing_estimate",
                    generated_text="Primary output"
                ),
                SubTask(
                    task_id="TASK-2",
                    description="Fallback subtask",
                    category="general",
                    assigned_model="openrouter",
                    provider="OpenRouter API",
                    execution_success=True,
                    cost=0.0008,
                    cost_source="configured_pricing_estimate",
                    generated_text="Fallback output"
                )
            ]

    class DummyScheduler:
        async def execute_plan_async(self, **kwargs):
            return kwargs["subtasks"]

    class DummyAggregator:
        def aggregate_results(self, **kwargs):
            return "Combined response"

    decomposer = __import__("app.services.complex.complex_task_decomposer", fromlist=["ComplexTaskDecomposer"]).ComplexTaskDecomposer(
        dynamic_decomposer=DummyDynamicDecomposer(),
        parallel_scheduler=DummyScheduler(),
        task_aggregator=DummyAggregator(),
    )

    plan = __import__("asyncio", fromlist=["run"]).run(
        decomposer.execute_complex_plan_async(prompt="Complex request", execution_mode="online")
    )

    assert plan.total_workflow_cost == 0.0033
    assert plan.synthesis_cost == 0.0
    assert plan.cost_source == "configured_pricing_estimate"
    assert plan.total_tokens == 0


def test_cost_metrics_api_endpoint():
    """Requirement 7: GET /api/v1/metrics/cost endpoint returns 200 with correct fields."""
    res1 = client.get("/api/v1/metrics/cost")
    assert res1.status_code == 200
    data1 = res1.json()
    assert "total_api_cost" in data1
    assert "currency" in data1

    res2 = client.get("/api/metrics/cost")
    assert res2.status_code == 200
    data2 = res2.json()
    assert "total_api_cost" in data2
