import logging
import time
from typing import Optional, List, Dict, Any, Set, Tuple

from app.schemas.complex import SubTask
from app.schemas.response import ResponseGenerationRequest, ResponseGenerationResponse

logger = logging.getLogger("orchestrator")

RETRYABLE_PATTERNS = [
    "429", "resource_exhausted", "quota", "rate limit", "too many requests",
    "500", "502", "503", "504", "server_error", "service unavailable", "bad gateway", "gateway timeout",
    "timed out", "timeout", "connection refused", "connection reset", "network failure", "dns failure",
    "unreachable", "temporarily unavailable"
]

NON_RETRYABLE_PATTERNS = [
    "invalid_prompt", "empty prompt", "safety refusal", "content policy", "malformed request"
]


def classify_failure(error_message: Optional[str], status_code: Optional[int] = None) -> Tuple[bool, str]:
    """
    Classifies a provider execution failure.
    Returns: (is_retryable: bool, failure_type: str)
    """
    if not error_message:
        return True, "unknown_error"

    err_lower = error_message.lower()

    # Quota / Rate limit
    if "429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
        failure_type = "quota_exhausted" if ("quota" in err_lower or "resource_exhausted" in err_lower) else "rate_limit"
        return True, failure_type

    # Server Errors
    if any(code in err_lower for code in ["500", "502", "503", "504", "service unavailable", "bad gateway", "internal server"]):
        return True, "server_error"

    # Timeout / Connection Errors
    if "timed out" in err_lower or "timeout" in err_lower or "connection refused" in err_lower or "network failure" in err_lower or "connection error" in err_lower:
        return True, "timeout"

    # Invalid Auth / Key missing
    if "401" in err_lower or "403" in err_lower or "not configured" in err_lower or "api_key" in err_lower:
        return True, "unconfigured"

    # Non-retryable
    if any(p in err_lower for p in NON_RETRYABLE_PATTERNS):
        return False, "non_retryable"

    # Default to retryable provider error
    return True, "provider_error"


class RequestProviderTracker:
    """
    Per-request provider health tracker for quota-aware cooldown.
    Tracks exhausted providers within the lifecycle of a single user request or complex plan.
    """

    def __init__(self):
        self.exhausted_providers: Dict[str, str] = {}  # provider_name -> failure_type

    def mark_provider_exhausted(self, provider_name: str, failure_type: str):
        if provider_name and provider_name not in self.exhausted_providers:
            logger.warning(f"[PROVIDER COOLDOWN] Marking provider '{provider_name}' as exhausted for current request (reason={failure_type}).")
            self.exhausted_providers[provider_name] = failure_type

    def is_provider_exhausted(self, provider_name: Optional[str]) -> bool:
        if not provider_name:
            return False
        return provider_name in self.exhausted_providers

    def get_exhausted_list(self) -> List[str]:
        return list(self.exhausted_providers.keys())
