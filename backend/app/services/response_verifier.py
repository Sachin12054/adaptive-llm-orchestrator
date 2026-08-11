import re
import time
import logging
from typing import List, Dict, Any, Optional

from app.schemas.verification import (
    VerificationRequest,
    VerificationResponse,
    VerificationStatusResponse
)

logger = logging.getLogger("orchestrator")

class ResponseVerifier:
    def __init__(self):
        pass

    def get_status(self) -> VerificationStatusResponse:
        return VerificationStatusResponse(
            status="ready",
            service="response_verifier",
            factual_verification_available=False
        )

    def _calculate_relevance(self, prompt: str, text: str) -> float:
        # Extract meaningful alphanumeric keywords (len >= 3)
        prompt_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", prompt.lower()))
        text_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))
        
        # Exclude common stop words
        stop_words = {"the", "and", "is", "for", "what", "how", "why", "with", "that", "this", "from", "are", "was"}
        prompt_keywords = prompt_words - stop_words
        
        if not prompt_keywords:
            return 0.50

        overlap = prompt_keywords.intersection(text_words)
        relevance_ratio = len(overlap) / len(prompt_keywords)

        # Baseline text presence and key term overlap score
        if len(text_words) >= 5 and relevance_ratio > 0.0:
            return min(1.0, max(0.40, round(0.40 + 0.60 * relevance_ratio, 4)))
        elif len(text_words) >= 3:
            return 0.50
        else:
            return 0.30

    def _evaluate_structure(self, text: str) -> Tuple_Structure:
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        issues = []
        if word_count < 2:
            issues.append("Response is extremely short (fewer than 2 words).")

        # Check for error leakage
        error_patterns = [r"\bexception\b", r"\btraceback\b", r"\berror:\b", r"\bfailed to generate\b"]
        for pat in error_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                issues.append(f"Response contains raw system error pattern matching '{pat}'.")

        completeness_score = min(1.0, max(0.20, word_count / 15.0))
        structural_quality = 1.0 if not issues else 0.40

        return round(float(structural_quality), 4), round(float(completeness_score), 4), issues

    def verify_response(self, request: VerificationRequest) -> VerificationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt text for response verification cannot be empty or contain only whitespace.")

        logger.info(f"ResponseVerifier evaluating output for model '{request.selected_model}'...")

        # Case 1: Unconfigured or Failed Execution
        if not request.generation_success or request.execution_status != "completed" or request.generated_text is None:
            t1 = time.perf_counter()
            return VerificationResponse(
                verified=False,
                verification_status="not_verifiable",
                prompt=request.prompt,
                selected_model=request.selected_model,
                response_present=False,
                relevance_score=0.0,
                completeness_score=0.0,
                structural_quality_score=0.0,
                factual_verification_status="not_verified",
                issues=["Generated response is unavailable because model execution did not complete."],
                verification_reasoning=[
                    f"Model execution status was '{request.execution_status}' (success={request.generation_success}).",
                    "Generated response text is unavailable.",
                    "Verification could not meaningfully proceed.",
                    "Factual correctness is not established by this baseline verifier."
                ],
                verification_latency_ms=round((t1 - t0) * 1000, 2)
            )

        # Case 2: Empty or Whitespace Response
        cleaned_text = request.generated_text.strip()
        if not cleaned_text:
            t1 = time.perf_counter()
            return VerificationResponse(
                verified=False,
                verification_status="empty_response",
                prompt=request.prompt,
                selected_model=request.selected_model,
                response_present=False,
                relevance_score=0.0,
                completeness_score=0.0,
                structural_quality_score=0.0,
                factual_verification_status="not_verified",
                issues=["Generated response contains only whitespace."],
                verification_reasoning=[
                    "Generated text is empty or contains only whitespace.",
                    "Response failed baseline structural presence checks.",
                    "Factual correctness is not established by this baseline verifier."
                ],
                verification_latency_ms=round((t1 - t0) * 1000, 2)
            )

        # Case 3: Structural & Relevance Evaluation for Generated Text
        struct_quality, completeness, structural_issues = self._evaluate_structure(cleaned_text)
        relevance_score = self._calculate_relevance(request.prompt, cleaned_text)

        is_verified = (struct_quality >= 0.50 and relevance_score >= 0.30 and len(structural_issues) == 0)
        verification_status = "verified_baseline" if is_verified else "failed_checks"

        reasoning = [
            f"Response text is present ({len(cleaned_text)} characters, {len(cleaned_text.split())} words).",
            f"Structural quality score: {struct_quality:.2f} | Completeness score: {completeness:.2f}.",
            f"Prompt relevance score: {relevance_score:.2f} (evaluated keyword/topic overlap).",
            "Factual correctness is not established by this baseline verifier."
        ]

        if structural_issues:
            reasoning.extend([f"Detected issue: {issue}" for issue in structural_issues])

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        return VerificationResponse(
            verified=is_verified,
            verification_status=verification_status,
            prompt=request.prompt,
            selected_model=request.selected_model,
            response_present=True,
            relevance_score=relevance_score,
            completeness_score=completeness,
            structural_quality_score=struct_quality,
            factual_verification_status="not_verified",
            issues=structural_issues,
            verification_reasoning=reasoning,
            verification_latency_ms=latency_ms
        )
