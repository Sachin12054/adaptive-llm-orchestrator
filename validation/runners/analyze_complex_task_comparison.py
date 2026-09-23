import os
import sys
import json
import re
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("complex_analysis")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "complex_task_comparison"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_raw_data():
    single_file = os.path.join(RAW_DIR, "single_model_results.jsonl")
    orch_file = os.path.join(RAW_DIR, "orchestrated_results.jsonl")

    single_recs = {}
    with open(single_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                single_recs[r["prompt_id"]] = r

    orch_recs = {}
    with open(orch_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                orch_recs[r["prompt_id"]] = r

    prompts = sorted(list(set(single_recs.keys()).intersection(set(orch_recs.keys()))))
    return single_recs, orch_recs, prompts

def bootstrap_diff_ci(diffs, num_bootstrap=2000, ci=95):
    boot_means = []
    rng = np.random.RandomState(42)
    n = len(diffs)
    for _ in range(num_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, (100 - ci) / 2.0)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return round(float(low), 4), round(float(high), 4)

def parse_sub_objectives(prompt_text: str) -> List[str]:
    phrases = re.split(r'[\,\;\.\?\!]', prompt_text)
    objs = []
    for p in phrases:
        p_clean = p.strip()
        words = p_clean.split()
        if len(words) >= 2:
            objs.append(p_clean)
    return objs if objs else [prompt_text]

def evaluate_objective_coverage(response_text: str, objectives: List[str]) -> Tuple[float, int, int, List[str]]:
    if not response_text or not response_text.strip():
        return 0.0, 0, len(objectives), objectives

    resp_lower = response_text.lower()
    satisfied = 0
    missing = []

    for obj in objectives:
        words = [w.lower() for w in obj.split() if len(w) > 4]
        if not words:
            words = [w.lower() for w in obj.split() if len(w) > 2]
            
        if any(w in resp_lower for w in words):
            satisfied += 1
        else:
            missing.append(obj)

    cov_pct = round((satisfied / len(objectives)) * 100.0, 2) if objectives else 100.0
    return cov_pct, satisfied, len(objectives), missing

def perform_paired_analysis(single_recs, orch_recs, prompts):
    paired_data = []

    q_s_list, q_o_list = [], []
    l_s_list, l_o_list = [], []
    cov_s_list, cov_o_list = [], []
    tok_s_list, tok_o_list = [], []

    speedup_list = []
    eff_list = []
    max_conc_list = []

    for pid in prompts:
        s = single_recs[pid]
        o = orch_recs[pid]

        prompt_text = s["original_prompt"]
        objs = parse_sub_objectives(prompt_text)

        text_s = s.get("response_text", "")
        text_o = o.get("response_text", "")

        cov_s, sat_s, tot_s, missing_s = evaluate_objective_coverage(text_s, objs)
        cov_o, sat_o, tot_o, missing_o = evaluate_objective_coverage(text_o, objs)

        rel_s = s.get("relevance_score", 0.85)
        rel_o = o.get("relevance_score", 0.85)

        # Rubric 0-5 Quality Score
        q_s = round(0.50 * rel_s * 5.0 + 0.50 * (cov_s / 100.0) * 5.0, 2)
        q_o = round(0.50 * rel_o * 5.0 + 0.50 * (cov_o / 100.0) * 5.0, 2)

        l_s, l_o = s["end_to_end_latency_ms"] / 1000.0, o["end_to_end_latency_ms"] / 1000.0 # sec
        tok_s, tok_o = s["total_tokens"], o["total_tokens"]

        speedup = o.get("parallel_speedup", 1.0)
        eff = o.get("parallel_efficiency", 1.0)
        max_conc = o.get("max_concurrency", 1)

        q_diff = q_o - q_s
        l_diff = l_o - l_s
        cov_diff = cov_o - cov_s

        q_s_list.append(q_s)
        q_o_list.append(q_o)
        l_s_list.append(l_s)
        l_o_list.append(l_o)
        cov_s_list.append(cov_s)
        cov_o_list.append(cov_o)
        tok_s_list.append(tok_s)
        tok_o_list.append(tok_o)

        speedup_list.append(speedup)
        eff_list.append(eff)
        max_conc_list.append(max_conc)

        paired_data.append({
            "prompt_id": pid,
            "category": s["category"],
            "quality_single": q_s,
            "quality_orchestrated": q_o,
            "quality_difference": round(q_diff, 2),
            "latency_single_sec": round(l_s, 2),
            "latency_orchestrated_sec": round(l_o, 2),
            "latency_difference_sec": round(l_diff, 2),
            "latency_change_pct": round((l_o - l_s) / l_s * 100.0, 1) if l_s > 0 else 0.0,
            "coverage_single_pct": cov_s,
            "coverage_orchestrated_pct": cov_o,
            "coverage_difference_pct": round(cov_diff, 1),
            "parallel_speedup": speedup,
            "parallel_efficiency": eff,
            "max_concurrency": max_conc,
            "tokens_single": tok_s,
            "tokens_orchestrated": tok_o,
            "tokens_difference": tok_o - tok_s
        })

    # Statistical Aggregations
    q_diffs = np.array(q_o_list) - np.array(q_s_list)
    l_diffs = np.array(l_o_list) - np.array(l_s_list)
    cov_diffs = np.array(cov_o_list) - np.array(cov_s_list)

    # Paired t-tests
    t_q, p_q = stats.ttest_rel(q_o_list, q_s_list)
    t_l, p_l = stats.ttest_rel(l_o_list, l_s_list)
    t_cov, p_cov = stats.ttest_rel(cov_o_list, cov_s_list)

    # Wilcoxon tests
    try:
        w_q, p_w_q = stats.wilcoxon(q_o_list, q_s_list)
    except Exception:
        p_w_q = 1.0
    try:
        w_l, p_w_l = stats.wilcoxon(l_o_list, l_s_list)
    except Exception:
        p_w_l = 1.0
    try:
        w_cov, p_w_cov = stats.wilcoxon(cov_o_list, cov_s_list)
    except Exception:
        p_w_cov = 1.0

    # Cohen's d
    d_q = np.mean(q_diffs) / np.std(q_diffs, ddof=1) if np.std(q_diffs, ddof=1) > 0 else 0.0
    d_l = np.mean(l_diffs) / np.std(l_diffs, ddof=1) if np.std(l_diffs, ddof=1) > 0 else 0.0
    d_cov = np.mean(cov_diffs) / np.std(cov_diffs, ddof=1) if np.std(cov_diffs, ddof=1) > 0 else 0.0

    # Efficiency Metrics
    q_per_sec_s = [q_s_list[i] / l_s_list[i] for i in range(len(prompts))]
    q_per_sec_o = [q_o_list[i] / l_o_list[i] for i in range(len(prompts))]

    summary = {
        "sample_size_n": len(prompts),
        "quality": {
            "mean_single": round(float(np.mean(q_s_list)), 4),
            "mean_orchestrated": round(float(np.mean(q_o_list)), 4),
            "mean_difference": round(float(np.mean(q_diffs)), 4),
            "median_difference": round(float(np.median(q_diffs)), 4),
            "std_difference": round(float(np.std(q_diffs, ddof=1)), 4),
            "bootstrap_95_ci": bootstrap_diff_ci(q_diffs),
            "ttest_p_value": float(p_q),
            "wilcoxon_p_value": float(p_w_q),
            "cohens_d": round(float(d_q), 4),
            "statistically_significant": bool(p_q < 0.05)
        },
        "latency_seconds": {
            "mean_single": round(float(np.mean(l_s_list)), 2),
            "mean_orchestrated": round(float(np.mean(l_o_list)), 2),
            "median_single": round(float(np.median(l_s_list)), 2),
            "median_orchestrated": round(float(np.median(l_o_list)), 2),
            "p95_single": round(float(np.percentile(l_s_list, 95)), 2),
            "p95_orchestrated": round(float(np.percentile(l_o_list, 95)), 2),
            "mean_difference": round(float(np.mean(l_diffs)), 2),
            "median_difference": round(float(np.median(l_diffs)), 2),
            "bootstrap_95_ci": bootstrap_diff_ci(l_diffs),
            "mean_latency_reduction_pct": round((float(np.mean(l_s_list)) - float(np.mean(l_o_list))) / float(np.mean(l_s_list)) * 100.0, 1),
            "ttest_p_value": float(p_l),
            "wilcoxon_p_value": float(p_w_l),
            "cohens_d": round(float(d_l), 4),
            "statistically_significant": bool(p_l < 0.05)
        },
        "objective_coverage": {
            "mean_single_pct": round(float(np.mean(cov_s_list)), 2),
            "mean_orchestrated_pct": round(float(np.mean(cov_o_list)), 2),
            "mean_difference_pct": round(float(np.mean(cov_diffs)), 2),
            "median_difference_pct": round(float(np.median(cov_diffs)), 2),
            "bootstrap_95_ci": bootstrap_diff_ci(cov_diffs),
            "ttest_p_value": float(p_cov),
            "wilcoxon_p_value": float(p_w_cov),
            "cohens_d": round(float(d_cov), 4),
            "statistically_significant": bool(p_cov < 0.05)
        },
        "parallel_execution": {
            "mean_parallel_speedup": round(float(np.mean(speedup_list)), 2),
            "median_parallel_speedup": round(float(np.median(speedup_list)), 2),
            "mean_parallel_efficiency": round(float(np.mean(eff_list)), 2),
            "mean_max_concurrency": round(float(np.mean(max_conc_list)), 2)
        },
        "efficiency": {
            "quality_per_second_single": round(float(np.mean(q_per_sec_s)), 4),
            "quality_per_second_orchestrated": round(float(np.mean(q_per_sec_o)), 4),
            "mean_tokens_single": round(float(np.mean(tok_s_list)), 1),
            "mean_tokens_orchestrated": round(float(np.mean(tok_o_list)), 1)
        }
    }

    return paired_data, summary

def generate_figures(summary, paired_data):
    # 1. Quality Comparison
    plt.figure(figsize=(7, 4.5))
    cats = ["Quality (0-5)", "Objective Coverage (%)"]
    single_vals = [summary["quality"]["mean_single"], summary["objective_coverage"]["mean_single_pct"] / 20.0]
    orch_vals = [summary["quality"]["mean_orchestrated"], summary["objective_coverage"]["mean_orchestrated_pct"] / 20.0]

    x = np.arange(len(cats))
    width = 0.35
    plt.bar(x - width/2, single_vals, width, label='Single Model (DeepSeek R1)', color='#1f77b4')
    plt.bar(x + width/2, orch_vals, width, label='Orchestrated Pipeline', color='#2ca02c')

    plt.ylabel('Score (Normalized to 5.0 Scale)')
    plt.title('Quality & Objective Coverage Comparison')
    plt.xticks(x, cats)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "quality_comparison.png"))
    plt.close()

    # 2. Latency Comparison
    plt.figure(figsize=(7, 4.5))
    lat_cats = ["Mean Latency (s)", "P50 Latency (s)", "P95 Latency (s)"]
    s_lats = [summary["latency_seconds"]["mean_single"], summary["latency_seconds"]["median_single"], summary["latency_seconds"]["p95_single"]]
    o_lats = [summary["latency_seconds"]["mean_orchestrated"], summary["latency_seconds"]["median_orchestrated"], summary["latency_seconds"]["p95_orchestrated"]]

    x = np.arange(len(lat_cats))
    plt.bar(x - width/2, s_lats, width, label='Single Model', color='#d62728')
    plt.bar(x + width/2, o_lats, width, label='Orchestrated Pipeline', color='#2ca02c')

    plt.ylabel('Latency (Seconds)')
    plt.title('End-to-End Latency Comparison')
    plt.xticks(x, lat_cats)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "latency_comparison.png"))
    plt.close()

    # 3. Objective Coverage
    plt.figure(figsize=(7, 4.5))
    s_cov = summary["objective_coverage"]["mean_single_pct"]
    o_cov = summary["objective_coverage"]["mean_orchestrated_pct"]
    plt.bar(["Single Model", "Orchestrated"], [s_cov, o_cov], color=['#1f77b4', '#2ca02c'], width=0.5)
    plt.ylabel('Objective Coverage (%)')
    plt.title('Mean Objective Coverage Percentage')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "objective_coverage.png"))
    plt.close()

    # 4. Parallel Speedup
    plt.figure(figsize=(7, 4.5))
    speedups = [d["parallel_speedup"] for d in paired_data]
    plt.hist(speedups, bins=8, color='#9467bd', edgecolor='black', alpha=0.8)
    plt.axvline(np.mean(speedups), color='red', linestyle='--', label=f'Mean Speedup ({np.mean(speedups):.2f}x)')
    plt.xlabel('Measured Parallel Speedup (T_seq / T_wall)')
    plt.ylabel('Number of Prompts')
    plt.title('Parallel Speedup Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "parallel_speedup.png"))
    plt.close()

    # 5. Resource Comparison
    plt.figure(figsize=(7, 4.5))
    res_cats = ["Quality / Sec", "Total Tokens (hundreds)"]
    s_res = [summary["efficiency"]["quality_per_second_single"], summary["efficiency"]["mean_tokens_single"] / 100.0]
    o_res = [summary["efficiency"]["quality_per_second_orchestrated"], summary["efficiency"]["mean_tokens_orchestrated"] / 100.0]

    x = np.arange(len(res_cats))
    plt.bar(x - width/2, s_res, width, label='Single Model', color='#ff7f0e')
    plt.bar(x + width/2, o_res, width, label='Orchestrated', color='#2ca02c')
    plt.xticks(x, res_cats)
    plt.ylabel('Metric Value')
    plt.title('Efficiency & Resource Utilization Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "resource_comparison.png"))
    plt.close()

    # 6. Quality Latency Tradeoff
    plt.figure(figsize=(7, 4.5))
    plt.scatter([summary["latency_seconds"]["mean_single"]], [summary["quality"]["mean_single"]], color='#1f77b4', s=150, label='Single Model (DeepSeek R1)')
    plt.scatter([summary["latency_seconds"]["mean_orchestrated"]], [summary["quality"]["mean_orchestrated"]], color='#2ca02c', s=150, label='Orchestrated Pipeline')

    plt.xlabel('Mean End-to-End Latency (Seconds)')
    plt.ylabel('Mean Quality Score (0-5 Rubric)')
    plt.title('Quality vs Latency Trade-Off Space')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "quality_latency_tradeoff.png"))
    plt.close()

def main():
    logger.info("Executing Statistical Analysis for RUN_3_COMPLEX_TASK_COMPARISON...")
    single_recs, orch_recs, prompts = load_raw_data()
    logger.info(f"Loaded {len(prompts)} matched paired prompt records.")

    paired_data, summary = perform_paired_analysis(single_recs, orch_recs, prompts)

    # Save paired results
    paired_file = os.path.join(PROCESSED_DIR, "paired_results.jsonl")
    with open(paired_file, "w", encoding="utf-8") as f:
        for item in paired_data:
            f.write(json.dumps(item) + "\n")

    # Save metric summary
    summary_file = os.path.join(PROCESSED_DIR, "metric_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Manifest
    manifest = {
        "experiment_id": "RUN_3_COMPLEX_TASK_COMPARISON",
        "dataset_prompts_n": len(prompts),
        "condition_s_model": "deepseek-r1-7b",
        "condition_o_engine": "ComplexTaskDecomposer + ParallelTaskScheduler",
        "status": "COMPLETED_EMPIRICAL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    manifest_file = os.path.join(BASE_DIR, "complex_task_comparison_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generating publication figures...")
    generate_figures(summary, paired_data)

    logger.info("Statistical Analysis Complete 100%!")

if __name__ == "__main__":
    import time
    main()
