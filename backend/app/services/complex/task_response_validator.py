import hashlib
import re
import logging
from typing import Tuple, List, Set, Optional

logger = logging.getLogger("orchestrator")

# Common English stop words to ignore during keyword extraction
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "upon", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "would", "could", "ought", "subtask", "task", "objective", "description",
    "design", "explain", "create", "provide", "identify", "propose", "system", "structure",
    "architecture", "plan", "details", "solution", "section", "part"
}

# Generic Domain Signature Patterns & Off-Target Schema Identifiers
# Format: (domain_context_regex, prohibited_mismatched_domain_terms, rule_name)
GENERIC_DOMAIN_CONSISTENCY_RULES: List[Tuple[str, List[str], str]] = [
    (
        r"\b(campus|university|college|student|academic|faculty|school|amrita|hostel|dorm|canteen|classroom)\b",
        ["books", "authors", "loans", "borrower", "borrowers", "isbn", "genres", "publisher", "shelf"],
        "Library Management schema detected for university/campus request"
    ),
    (
        r"\b(campus|university|college|student|academic|faculty|school|amrita|hospital|clinical|patient|doctor)\b",
        ["shopping cart", "checkout", "sku", "product catalog", "shipping address", "item stock", "merchant"],
        "E-commerce schema detected for education/campus request"
    ),
    (
        r"\b(campus|university|college|student|academic|faculty|school|amrita)\b",
        ["cryptocurrency", "token sale", "blockchain mining", "nft collection", "crypto wallet"],
        "Cryptocurrency schema detected for education/campus request"
    ),
    (
        r"\b(e-commerce|shopping|retail|store|cart|merchant)\b",
        ["patient id", "diagnosis", "prescription", "medical record", "doctor notes"],
        "Medical schema detected for e-commerce request"
    )
]

class TaskResponseValidator:
    """
    Validates task prompt immutability via SHA-256 prompt hashing
    and generic semantic domain-consistency task-response relevance.
    """

    @staticmethod
    def compute_prompt_hash(task_id: str, original_user_prompt: Optional[str], task_objective: str) -> str:
        """
        Computes a deterministic SHA-256 hash of normalized task_id, original_user_prompt, and task_objective.
        """
        orig_clean = (original_user_prompt or "").strip().lower()
        obj_clean = task_objective.strip().lower()
        t_clean = task_id.strip().upper()
        normalized = f"{t_clean}:{orig_clean}:{obj_clean}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def validate_prompt_integrity(
        expected_hash: str,
        task_id: str,
        original_user_prompt: Optional[str],
        task_objective: str
    ) -> bool:
        """
        Asserts that current prompt details hash to expected_hash.
        Raises ValueError on hash mismatch to prevent prompt contamination.
        """
        if not expected_hash:
            return True
            
        current_hash = TaskResponseValidator.compute_prompt_hash(task_id, original_user_prompt, task_objective)
        if current_hash != expected_hash:
            err_msg = (
                f"[PROMPT CONTAMINATION DETECTED] Task {task_id}: Prompt hash mismatch! "
                f"Expected '{expected_hash[:12]}...', got '{current_hash[:12]}...'."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return True

    @staticmethod
    def validate_response_relevance(
        task_id: str,
        original_user_prompt: Optional[str],
        task_objective: str,
        category: str,
        response_text: str
    ) -> Tuple[bool, float, str]:
        """
        Performs generic semantic domain-consistency and lexical relevance validation.
        Evaluates ORIGINAL USER REQUEST + SUBTASK OBJECTIVE + ACTUAL MODEL RESPONSE.
        Returns:
            (is_valid: bool, relevance_score: float, reasoning: str)
        """
        if not response_text or not response_text.strip():
            return False, 0.0, "Response text is empty or whitespace only"

        resp_clean = response_text.strip()
        resp_lower = resp_clean.lower()
        obj_lower = task_objective.lower()
        orig_lower = (original_user_prompt or "").lower()
        full_context_lower = f"{orig_lower} {obj_lower}"

        # 1. Generic Domain Consistency Check
        for domain_regex, prohibited_terms, rule_name in GENERIC_DOMAIN_CONSISTENCY_RULES:
            # Check if global context or subtask specifies Target Domain A
            if re.search(domain_regex, full_context_lower):
                # Count presence of off-target Mismatched Domain B vocabulary
                mismatched_hits = [t for t in prohibited_terms if re.search(r"\b" + re.escape(t) + r"\b", resp_lower)]
                
                # Check if target domain keywords are absent from the response while off-target terms dominate
                target_domain_hits = re.findall(domain_regex, resp_lower)
                
                if len(mismatched_hits) >= 2 and len(target_domain_hits) == 0:
                    reason = (
                        f"Off-target generic domain response detected: response features mismatched domain vocabulary "
                        f"({', '.join(mismatched_hits)}) without addressing original request domain concepts ({rule_name})."
                    )
                    logger.warning(f"[DOMAIN MISMATCH FAILURE] Task {task_id}: {reason}")
                    return False, 0.10, reason

        # 2. Extract Significant Domain Keywords from Global Request Context + Subtask Objective
        orig_words = re.findall(r"\b[a-z]{3,}\b", orig_lower)
        obj_words = re.findall(r"\b[a-z]{3,}\b", obj_lower)
        
        domain_keywords = [w for w in orig_words if w not in STOP_WORDS]
        obj_keywords = [w for w in obj_words if w not in STOP_WORDS]
        
        all_target_keywords = set(domain_keywords + obj_keywords)

        if not all_target_keywords:
            return True, 1.0, "Objective and request contain no specific domain keywords"

        matched_count = sum(1 for kw in all_target_keywords if kw in resp_lower)
        overlap_ratio = matched_count / float(len(all_target_keywords))
        relevance_score = round(min(1.0, max(0.0, overlap_ratio * 1.5)), 4)

        # 3. Low Keyword Overlap Safeguard for Substantial Responses
        word_count = len(resp_clean.split())
        if word_count >= 40 and overlap_ratio < 0.06:
            reason = f"Low semantic relevance score ({relevance_score:.2f}); response does not address target domain keywords ({matched_count}/{len(all_target_keywords)} matched)."
            logger.warning(f"[RELEVANCE WARNING] Task {task_id}: {reason}")
            return False, relevance_score, reason

        return True, max(0.60, relevance_score), f"Response passed semantic relevance validation (score: {relevance_score:.2f})."
