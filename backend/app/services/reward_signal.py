import time
import logging
from typing import List, Dict, Any, Optional

from app.schemas.reward import (
    RewardComputeRequest,
    RewardComputeResponse,
    RewardStatusResponse,
    RewardBreakdown,
    ComponentContribution
)

logger = logging.getLogger("orchestrator")

class RewardSignal:
    def __init__(self):
        pass

    def get_status(self) -> RewardStatusResponse:
        return RewardStatusResponse(
            status="ready",
            service="reward_signal",
            learning_active=False
        )

    def compute_reward(self, request: RewardComputeRequest) -> RewardComputeResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt text for reward computation cannot be empty or contain only whitespace.")

        logger.info(f"RewardSignal computing baseline reward for model '{request.selected_model}'...")

        # 1. Component Scores
        quality_score = request.structural_quality_score if request.response_present else 0.0
        completeness_score = request.completeness_score if request.response_present else 0.0
        relevance_score = request.relevance_score if request.response_present else 0.0
        verification_score = 1.0 if request.verified else 0.0
        execution_score = 1.0 if (request.execution_success and request.execution_status == "completed") else 0.0

        # 2. Component Weights
        quality_weight = 0.30
        completeness_weight = 0.20
        relevance_weight = 0.25
        verification_weight = 0.15
        execution_weight = 0.10

        # 3. Contributions
        quality_contrib = round(quality_score * quality_weight, 4)
        completeness_contrib = round(completeness_score * completeness_weight, 4)
        relevance_contrib = round(relevance_score * relevance_weight, 4)
        verification_contrib = round(verification_score * verification_weight, 4)
        execution_contrib = round(execution_score * execution_weight, 4)

        # 4. Total Reward Calculation
        raw_reward = (
            quality_contrib +
            completeness_contrib +
            relevance_contrib +
            verification_contrib +
            execution_contrib
        )
        clamped_reward = round(float(min(1.0, max(0.0, raw_reward))), 4)

        # 5. Reward Status
        if not request.execution_success or request.execution_status != "completed":
            reward_status = "failed_execution"
        elif request.verified:
            reward_status = "completed_verified"
        elif request.response_present:
            reward_status = "completed_unverified"
        else:
            reward_status = "not_verifiable"

        # 6. Reasoning Bullets
        reasoning = []
        if request.execution_success and request.execution_status == "completed":
            reasoning.append(f"Model execution completed successfully (execution component contributed {execution_contrib:.4f}).")
        else:
            reasoning.append(f"Model execution status was '{request.execution_status}' (execution component contributed {execution_contrib:.4f}).")

        if request.response_present:
            reasoning.append(f"Structural quality score {quality_score:.2f} contributed {quality_contrib:.4f} (weight={quality_weight}).")
            reasoning.append(f"Completeness score {completeness_score:.2f} contributed {completeness_contrib:.4f} (weight={completeness_weight}).")
            reasoning.append(f"Prompt relevance score {relevance_score:.2f} contributed {relevance_contrib:.4f} (weight={relevance_weight}).")
        else:
            reasoning.append("No response was present for meaningful quality/relevance evaluation.")

        if request.verified:
            reasoning.append(f"Baseline verification passed (verification component contributed {verification_contrib:.4f}).")
        else:
            reasoning.append(f"Baseline verification failed or not verifiable (verification status: '{request.verification_status}').")

        reasoning.append("Factual correctness was not established because factual verification is inactive.")
        reasoning.append("This is a deterministic baseline reward signal for future RL learning. It is NOT a trained RL policy.")
        reasoning.append(f"Final normalized reward signal: {clamped_reward:.4f}.")

        breakdown = RewardBreakdown(
            quality=ComponentContribution(score=quality_score, weight=quality_weight, contribution=quality_contrib),
            completeness=ComponentContribution(score=completeness_score, weight=completeness_weight, contribution=completeness_contrib),
            relevance=ComponentContribution(score=relevance_score, weight=relevance_weight, contribution=relevance_contrib),
            verification=ComponentContribution(score=verification_score, weight=verification_weight, contribution=verification_contrib),
            execution=ComponentContribution(score=execution_score, weight=execution_weight, contribution=execution_contrib)
        )

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        return RewardComputeResponse(
            success=True,
            selected_model=request.selected_model,
            reward=clamped_reward,
            reward_breakdown=breakdown,
            reward_status=reward_status,
            reasoning=reasoning,
            factual_verification_status="not_verified",
            latency_ms=latency_ms
        )
