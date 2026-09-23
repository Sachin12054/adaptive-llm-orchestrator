import time
import logging
from typing import List, Dict, Any, Tuple, Optional

from app.schemas.complex import SubTask
from app.services.response_generator import ResponseGenerator
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.response import ResponseGenerationRequest
from app.schemas.decision import DecisionRequest

logger = logging.getLogger("orchestrator")

def deterministic_s2_assembly(
    original_prompt: str,
    subtasks: List[SubTask]
) -> Tuple[str, float]:
    """
    S2 Strategy: Zero-LLM Deterministic Structured Assembly.
    - Preserves all completed subtask outputs, ordering, code blocks, and tables.
    - Generates markdown section headers and an executive table of contents.
    - Guaranteed 100% byte-for-byte reproducible.
    - Zero synthesis LLM token generation.
    """
    t0 = time.perf_counter()
    completed = [t for t in subtasks if t.execution_success and t.generated_text]
    failed = [t for t in subtasks if not t.execution_success]

    if not completed:
        return f"# Complex Task Execution Failed\n\nNo subtask outputs available across {len(subtasks)} subtasks.", 0.0

    lines = []
    lines.append("# Executive Synthesis & Modular Analysis")
    lines.append("")
    lines.append(f"**Target System Prompt**: {original_prompt.strip()}")
    lines.append("")

    if failed:
        lines.append(
            f"> [!WARNING]\n"
            f"> **Partial Execution Notice**: {len(completed)} of {len(subtasks)} subtasks completed successfully. "
            f"{len(failed)} subtask(s) failed or were unavailable."
        )
        lines.append("")

    # Section Index / Overview
    lines.append("### Task Component Index")
    lines.append("| Task ID | Focus / Objective | Assigned Model | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for t in subtasks:
        obj_short = (t.task_objective or t.description or "Subtask").replace("|", "-")
        if len(obj_short) > 60:
            obj_short = obj_short[:57] + "..."
        model_name = getattr(t, 'assigned_model', getattr(t, 'selected_model', 'local_model'))
        status_str = "Completed" if (t.execution_success and t.generated_text) else "Unavailable"
        lines.append(f"| {t.task_id} | {obj_short} | `{model_name}` | {status_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Subtask Body Sections
    for t in subtasks:
        lines.append(f"## Subtask [{t.task_id}]: {t.task_objective or t.description}")
        lines.append("")
        if t.execution_success and t.generated_text:
            lines.append(t.generated_text.strip())
        else:
            lines.append(f"*[Execution Unavailable: {t.error_message or 'Provider error'}]*")
        lines.append("")
        lines.append("---")
        lines.append("")

    final_text = "\n".join(lines).strip()
    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000, 2)
    return final_text, latency_ms

def hybrid_s3_assembly(
    original_prompt: str,
    subtasks: List[SubTask],
    response_generator: ResponseGenerator,
    execution_mode: str = "local"
) -> Tuple[str, float, float, bool, Optional[Any]]:
    """
    S3 Strategy: Hybrid Assembly (S2 Deterministic Assembly + Gemma 3 4B Executive Summary).
    Returns (final_text, assembly_latency_ms, summary_latency_ms, fallback_used, synthesis_response).
    """
    s2_text, asm_ms = deterministic_s2_assembly(original_prompt, subtasks)

    completed = [t for t in subtasks if t.execution_success and t.generated_text]
    if not completed:
        return s2_text, asm_ms, 0.0, True, None

    # Build summary prompt from subtask objectives
    bullet_list = "\n".join([f"- Task {t.task_id}: {t.task_objective or t.description}" for t in completed])
    summary_prompt = (
        f"Original User Prompt: {original_prompt}\n\n"
        f"Executed Subtasks:\n{bullet_list}\n\n"
        f"Instructions: Write a concise 2-sentence executive summary highlighting the main architectural components above. "
        f"Do NOT invent new facts. Keep output strictly under 80 words."
    )

    t0_sum = time.perf_counter()
    exec_mode = "online" if execution_mode.lower() in ["online", "mistral", "gemini"] else "local"
    sum_model = "gemma-3-4b" if exec_mode == "local" else "gemini-3.5-flash"

    gen_req = ResponseGenerationRequest(
        prompt=summary_prompt,
        selected_model=sum_model,
        execution_mode=exec_mode,
        max_output_tokens=150,
        temperature=0.3
    )

    try:
        gen_res = response_generator.generate_response(gen_req)
        t1_sum = time.perf_counter()
        sum_ms = round((t1_sum - t0_sum) * 1000, 2)

        if gen_res.success and gen_res.generated_text and gen_res.generated_text.strip():
            summary_clean = gen_res.generated_text.strip()
            # Prefix executive summary to S2 output
            hybrid_text = (
                f"# Executive Synthesis & Modular Analysis\n\n"
                f"> **Executive Summary**: {summary_clean}\n\n"
                f"{s2_text.replace('# Executive Synthesis & Modular Analysis', '').strip()}"
            )
            return hybrid_text, asm_ms, sum_ms, False, gen_res
        else:
            return s2_text, asm_ms, sum_ms, True, gen_res
    except Exception as e:
        logger.warning(f"S3 Summary generation failed: {str(e)}. Falling back to S2.")
        return s2_text, asm_ms, 0.0, True, None

def llm_synthesis_s0_s1(
    original_prompt: str,
    subtasks: List[SubTask],
    synth_model: str,
    response_generator: ResponseGenerator,
    execution_mode: str = "local"
) -> Tuple[str, float, Optional[Any]]:
    """
    S0 / S1 Strategy: Full LLM Synthesis.
    - S0 uses deepseek-r1-7b
    - S1 uses gemma-3-4b
    """
    t0 = time.perf_counter()
    completed = [t for t in subtasks if t.execution_success and t.generated_text]
    failed = [t for t in subtasks if not t.execution_success]

    if not completed:
        return f"Complex task execution failed across all {len(subtasks)} subtasks.", 0.0, None

    sections = []
    if failed:
        sections.append(
            f"**[PARTIAL EXECUTION NOTICE]**: {len(completed)} of {len(subtasks)} subtasks completed successfully."
        )

    for t in subtasks:
        if t.execution_success and t.generated_text:
            text_snippet = t.generated_text.strip()
            if len(text_snippet) > 1500:
                text_snippet = text_snippet[:1500] + "... [truncated for synthesis]"
            sections.append(f"### Subtask [{t.task_id}]: {t.task_objective or t.description}\n{text_snippet}")
        else:
            sections.append(f"### Subtask [{t.task_id}]: {t.task_objective or t.description}\n*[Execution Unavailable]*")

    subtasks_combined_text = "\n\n".join(sections)
    synth_prompt = (
        f"User Prompt: {original_prompt}\n\n"
        f"Below are the verified outputs from decomposed subtasks:\n\n"
        f"{subtasks_combined_text}\n\n"
        f"Instructions: Synthesize all available subtask results into a clean, cohesive, and comprehensive response. "
        f"Preserve all specific facts, numbers, code, and recommendations. Clearly indicate any missing or incomplete areas if subtasks failed. Do not add fake information."
    )

    if len(synth_prompt) > 7500:
        synth_prompt = synth_prompt[:7500] + "\n\nInstructions: Synthesize the above subtask outputs."

    exec_mode = "online" if execution_mode.lower() in ["online", "mistral", "gemini"] else "local"

    gen_req = ResponseGenerationRequest(
        prompt=synth_prompt,
        selected_model=synth_model,
        execution_mode=exec_mode,
        max_output_tokens=1000,
        temperature=0.7
    )

    gen_res = response_generator.generate_response(gen_req)
    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000, 2)

    if gen_res.success and gen_res.generated_text and gen_res.generated_text.strip():
        return gen_res.generated_text.strip(), latency_ms, gen_res

    # Fallback to deterministic format if LLM synthesis call fails
    fallback_text, _ = deterministic_s2_assembly(original_prompt, subtasks)
    return fallback_text, latency_ms, gen_res
