import os
import sys
import json
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def ci95(data):
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "std": 0.0, "std_err": 0.0, "margin": 0.0, "ci_lower": mean, "ci_upper": mean, "p50": mean, "p95": mean}
    std_err = float(np.std(arr, ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n-1))
    margin = float(t_crit * std_err)
    return {
        "mean": round(mean, 4),
        "std": round(float(np.std(arr, ddof=1)), 4),
        "std_err": round(std_err, 4),
        "margin": round(margin, 4),
        "ci_lower": round(mean - margin, 4),
        "ci_upper": round(mean + margin, 4),
        "p50": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4)
    }

def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    n = len(diff)
    if n <= 1:
        return 0.0
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0.0
    return float(np.mean(diff) / sd_diff)

def analyze():
    base_dir = 'validation/results/complex_task_synthesis_optimization'
    raw_file = os.path.join(base_dir, 'raw', 'synthesis_ablation_results.jsonl')
    processed_dir = os.path.join(base_dir, 'processed')
    figures_dir = os.path.join(base_dir, 'figures')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    with open(raw_file, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(records)} strategy trial records.")

    # Group by strategy_id and prompt_id
    by_strat = {"S0": {}, "S1": {}, "S2": {}, "S3": {}}
    for r in records:
        sid = r["strategy_id"]
        pid = r["prompt_id"]
        by_strat[sid][pid] = r

    prompt_ids = list(by_strat["S0"].keys())
    n = len(prompt_ids)
    print(f"Total paired prompt records per strategy: N = {n}")

    strats = ["S0", "S1", "S2", "S3"]
    metrics_by_strat = {s: {
        "synth_latency": [],
        "e2e_latency": [],
        "quality": [],
        "coverage": [],
        "tokens": [],
        "chars": [],
        "cpu": [],
        "ram": [],
        "quality_per_sec": [],
        "coverage_per_sec": []
    } for s in strats}

    for pid in prompt_ids:
        for s in strats:
            rec = by_strat[s][pid]
            s_lat = rec["synthesis_latency_ms"] / 1000.0
            e_lat = rec["total_e2e_latency_ms"] / 1000.0
            q = rec["quality_score_0_to_5"]
            c = rec["objective_coverage_pct"]
            tok = rec["output_tokens"]
            ch = rec["character_count"]
            cpu = rec["cpu_utilization_pct"]
            ram = rec["ram_usage_mb"]

            qps = q / e_lat if e_lat > 0 else 0.0
            cps = c / e_lat if e_lat > 0 else 0.0

            metrics_by_strat[s]["synth_latency"].append(s_lat)
            metrics_by_strat[s]["e2e_latency"].append(e_lat)
            metrics_by_strat[s]["quality"].append(q)
            metrics_by_strat[s]["coverage"].append(c)
            metrics_by_strat[s]["tokens"].append(tok)
            metrics_by_strat[s]["chars"].append(ch)
            metrics_by_strat[s]["cpu"].append(cpu)
            metrics_by_strat[s]["ram"].append(ram)
            metrics_by_strat[s]["quality_per_sec"].append(qps)
            metrics_by_strat[s]["coverage_per_sec"].append(cps)

    # Compute summary stats for each strategy
    summary_stats = {}
    for s in strats:
        summary_stats[s] = {
            "synthesis_latency_s": ci95(metrics_by_strat[s]["synth_latency"]),
            "e2e_latency_s": ci95(metrics_by_strat[s]["e2e_latency"]),
            "quality": ci95(metrics_by_strat[s]["quality"]),
            "coverage_pct": ci95(metrics_by_strat[s]["coverage"]),
            "output_tokens": ci95(metrics_by_strat[s]["tokens"]),
            "character_count": ci95(metrics_by_strat[s]["chars"]),
            "cpu_utilization_pct": ci95(metrics_by_strat[s]["cpu"]),
            "ram_usage_mb": ci95(metrics_by_strat[s]["ram"]),
            "quality_per_sec": ci95(metrics_by_strat[s]["quality_per_sec"]),
            "coverage_per_sec": ci95(metrics_by_strat[s]["coverage_per_sec"])
        }

    # Paired comparisons: S0 vs S1, S0 vs S2, S0 vs S3
    paired_comparisons = {}
    for target in ["S1", "S2", "S3"]:
        # Quality test
        q0 = metrics_by_strat["S0"]["quality"]
        qt = metrics_by_strat[target]["quality"]
        q_ttest = stats.ttest_rel(qt, q0)
        q_wilc = stats.wilcoxon(np.array(qt) - np.array(q0))
        q_d = cohens_d_paired(qt, q0)

        # Coverage test
        c0 = metrics_by_strat["S0"]["coverage"]
        ct = metrics_by_strat[target]["coverage"]
        c_ttest = stats.ttest_rel(ct, c0)
        c_wilc = stats.wilcoxon(np.array(ct) - np.array(c0))
        c_d = cohens_d_paired(ct, c0)

        # Synthesis Latency test
        sl0 = metrics_by_strat["S0"]["synth_latency"]
        slt = metrics_by_strat[target]["synth_latency"]
        sl_ttest = stats.ttest_rel(slt, sl0)
        sl_wilc = stats.wilcoxon(np.array(slt) - np.array(sl0))
        sl_d = cohens_d_paired(slt, sl0)
        synth_reduction_pct = ((np.mean(sl0) - np.mean(slt)) / np.mean(sl0)) * 100.0

        # E2E Latency test
        el0 = metrics_by_strat["S0"]["e2e_latency"]
        elt = metrics_by_strat[target]["e2e_latency"]
        el_ttest = stats.ttest_rel(elt, el0)
        el_wilc = stats.wilcoxon(np.array(elt) - np.array(el0))
        el_d = cohens_d_paired(elt, el0)
        e2e_reduction_pct = ((np.mean(el0) - np.mean(elt)) / np.mean(el0)) * 100.0

        paired_comparisons[f"S0_vs_{target}"] = {
            "synthesis_latency_reduction_pct": round(float(synth_reduction_pct), 2),
            "e2e_latency_reduction_pct": round(float(e2e_reduction_pct), 2),
            "quality": {
                "mean_diff": round(float(np.mean(qt) - np.mean(q0)), 4),
                "t_statistic": round(float(q_ttest.statistic), 4),
                "p_value": float(q_ttest.pvalue),
                "wilcoxon_p_value": float(q_wilc.pvalue),
                "cohens_d": round(q_d, 4),
                "statistically_significant": bool(q_ttest.pvalue < 0.05)
            },
            "coverage": {
                "mean_diff_pct": round(float(np.mean(ct) - np.mean(c0)), 4),
                "t_statistic": round(float(c_ttest.statistic), 4),
                "p_value": float(c_ttest.pvalue),
                "wilcoxon_p_value": float(c_wilc.pvalue),
                "cohens_d": round(c_d, 4),
                "statistically_significant": bool(c_ttest.pvalue < 0.05)
            },
            "synthesis_latency": {
                "mean_diff_s": round(float(np.mean(slt) - np.mean(sl0)), 4),
                "t_statistic": round(float(sl_ttest.statistic), 4),
                "p_value": float(sl_ttest.pvalue),
                "wilcoxon_p_value": float(sl_wilc.pvalue),
                "cohens_d": round(sl_d, 4),
                "statistically_significant": bool(sl_ttest.pvalue < 0.05)
            },
            "e2e_latency": {
                "mean_diff_s": round(float(np.mean(elt) - np.mean(el0)), 4),
                "t_statistic": round(float(el_ttest.statistic), 4),
                "p_value": float(el_ttest.pvalue),
                "wilcoxon_p_value": float(el_wilc.pvalue),
                "cohens_d": round(el_d, 4),
                "statistically_significant": bool(el_ttest.pvalue < 0.05)
            }
        }

    # Save summary json
    audit_results = {
        "metadata": {
            "experiment": "RUN_4_SYNTHESIS_OPTIMIZATION",
            "title": "Controlled Synthesis Ablation Results",
            "sample_size_n": n,
            "strategies_evaluated": ["S0", "S1", "S2", "S3"],
            "execution_environment": "Local Ollama (CPU 16 Threads)",
            "no_rerun_compliance": True
        },
        "summary_statistics": summary_stats,
        "paired_comparisons": paired_comparisons
    }

    out_file = os.path.join(processed_dir, "metric_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Saved metric summary to {out_file}")

    # GENERATE FIGURES (1-8)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Fig 1: Synthesis Latency by Strategy
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [summary_stats[s]["synthesis_latency_s"]["mean"] for s in strats]
    errors = [summary_stats[s]["synthesis_latency_s"]["margin"] for s in strats]
    bars = ax.bar(strats, means, yerr=errors, capsize=5, color=['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8'])
    ax.set_ylabel("Synthesis Latency (seconds)")
    ax.set_title("Figure 1: Synthesis Latency by Strategy (Mean ± 95% CI)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "synthesis_latency_by_strategy.png"), dpi=300)
    plt.close()

    # Fig 2: Quality by Strategy
    fig, ax = plt.subplots(figsize=(8, 5))
    q_means = [summary_stats[s]["quality"]["mean"] for s in strats]
    q_errs = [summary_stats[s]["quality"]["margin"] for s in strats]
    bars = ax.bar(strats, q_means, yerr=q_errs, capsize=5, color=['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8'])
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Response Quality Score (0-5 Rubric)")
    ax.set_title("Figure 2: Response Quality by Strategy (Mean ± 95% CI)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "quality_by_strategy.png"), dpi=300)
    plt.close()

    # Fig 3: Objective Coverage by Strategy
    fig, ax = plt.subplots(figsize=(8, 5))
    c_means = [summary_stats[s]["coverage_pct"]["mean"] for s in strats]
    c_errs = [summary_stats[s]["coverage_pct"]["margin"] for s in strats]
    bars = ax.bar(strats, c_means, yerr=c_errs, capsize=5, color=['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8'])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Objective Coverage (%)")
    ax.set_title("Figure 3: Objective Coverage by Strategy (Mean ± 95% CI)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "objective_coverage_by_strategy.png"), dpi=300)
    plt.close()

    # Fig 4: Quality vs Latency Pareto Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    e2e_means = [summary_stats[s]["e2e_latency_s"]["mean"] for s in strats]
    colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8']
    for i, s in enumerate(strats):
        ax.scatter(e2e_means[i], q_means[i], color=colors[i], s=120, label=f"{s}")
        ax.annotate(f"{s} ({e2e_means[i]:.1f}s, {q_means[i]:.2f})", (e2e_means[i]+1, q_means[i]+0.03), fontweight='bold')
    ax.set_xlabel("Mean Total E2E Latency (seconds)")
    ax.set_ylabel("Mean Quality Score (0-5 Rubric)")
    ax.set_title("Figure 4: Quality vs Latency Pareto Frontier")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "quality_vs_latency_pareto.png"), dpi=300)
    plt.close()

    # Fig 5: Latency Breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    decomp_mean = np.mean([by_strat["S0"][pid]["decomposition_latency_ms"]/1000.0 for pid in prompt_ids])
    subtask_mean = np.mean([by_strat["S0"][pid]["subtask_parallel_wall_time_ms"]/1000.0 for pid in prompt_ids])
    synth_means = [summary_stats[s]["synthesis_latency_s"]["mean"] for s in strats]
    
    p1 = ax.bar(strats, [decomp_mean]*4, label="Decomposition", color='#cccccc')
    p2 = ax.bar(strats, [subtask_mean]*4, bottom=[decomp_mean]*4, label="Subtask Parallel Exec", color='#5bc0de')
    p3 = ax.bar(strats, synth_means, bottom=[decomp_mean+subtask_mean]*4, label="Synthesis Stage", color=['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8'])

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Figure 5: Total End-to-End Latency Breakdown by Pipeline Stage")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "latency_breakdown.png"), dpi=300)
    plt.close()

    # Fig 6: Output Token Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    tok_means = [summary_stats[s]["output_tokens"]["mean"] for s in strats]
    bars = ax.bar(strats, tok_means, color=['#d9534f', '#f0ad4e', '#5cb85c', '#0275d8'])
    ax.set_ylabel("Output Tokens Generated")
    ax.set_title("Figure 6: Mean Output Tokens Generated by Strategy")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "output_token_comparison.png"), dpi=300)
    plt.close()

    # Fig 7: Quality Difference S0 vs S2
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_s2 = np.array(metrics_by_strat["S2"]["quality"]) - np.array(metrics_by_strat["S0"]["quality"])
    ax.hist(diff_s2, bins=7, color='#5cb85c', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label="Zero Difference")
    ax.axvline(np.mean(diff_s2), color='darkgreen', linestyle='-', linewidth=2, label=f"Mean Diff ({np.mean(diff_s2):.2f})")
    ax.set_xlabel("Quality Score Difference (S2 - S0)")
    ax.set_ylabel("Prompt Count")
    ax.set_title("Figure 7: Paired Quality Score Difference Distribution (S2 vs S0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "quality_difference_S0_vs_S2.png"), dpi=300)
    plt.close()

    # Fig 8: Quality Difference S0 vs S3
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_s3 = np.array(metrics_by_strat["S3"]["quality"]) - np.array(metrics_by_strat["S0"]["quality"])
    ax.hist(diff_s3, bins=7, color='#0275d8', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label="Zero Difference")
    ax.axvline(np.mean(diff_s3), color='navy', linestyle='-', linewidth=2, label=f"Mean Diff ({np.mean(diff_s3):.2f})")
    ax.set_xlabel("Quality Score Difference (S3 - S0)")
    ax.set_ylabel("Prompt Count")
    ax.set_title("Figure 8: Paired Quality Score Difference Distribution (S3 vs S0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "quality_difference_S0_vs_S3.png"), dpi=300)
    plt.close()

    print(f"Generated all 8 publication figures in {figures_dir}")

if __name__ == '__main__':
    analyze()
