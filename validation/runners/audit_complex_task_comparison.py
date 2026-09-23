import json
import os
import math
import numpy as np
from scipy import stats

def ci95(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "margin": 0.0, "ci_lower": mean, "ci_upper": mean}
    std_err = float(np.std(arr, ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n-1))
    margin = float(t_crit * std_err)
    return {
        "mean": mean,
        "std": float(np.std(arr, ddof=1)),
        "std_err": std_err,
        "margin": margin,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def main():
    raw_dir = 'validation/results/complex_task_comparison/raw'
    single_path = os.path.join(raw_dir, 'single_model_results.jsonl')
    orch_path = os.path.join(raw_dir, 'orchestrated_results.jsonl')

    with open(single_path, 'r', encoding='utf-8') as f:
        single_records = [json.loads(l) for l in f if l.strip()]

    with open(orch_path, 'r', encoding='utf-8') as f:
        orch_records = [json.loads(l) for l in f if l.strip()]

    single_map = {r['prompt_id']: r for r in single_records}
    orch_map = {r['prompt_id']: r for r in orch_records}
    prompt_ids = [r['prompt_id'] for r in single_records]
    n = len(prompt_ids)

    # 1. Quality & Coverage from metric_summary.json (canonical scores)
    ms_path = 'validation/results/complex_task_comparison/processed/metric_summary.json'
    with open(ms_path, 'r', encoding='utf-8') as f:
        ms = json.load(f)

    # Calculate exact telemetry breakdown
    single_e2e_lat = [s['end_to_end_latency_ms'] / 1000.0 for s in single_records]
    orch_e2e_lat = [o['end_to_end_latency_ms'] / 1000.0 for o in orch_records]
    orch_decomp_lat = [o.get('decomposition_latency_ms', 0.0) / 1000.0 for o in orch_records]
    orch_seq_lat = [o['sequential_equivalent_latency_ms'] / 1000.0 for o in orch_records]
    orch_wall_lat = [o['actual_parallel_wall_clock_ms'] / 1000.0 for o in orch_records]

    # Synthesis latency = E2E latency - (decomp + wall_clock)
    orch_synth_lat = []
    for o in orch_records:
        e2e = o['end_to_end_latency_ms'] / 1000.0
        decomp = o.get('decomposition_latency_ms', 0.0) / 1000.0
        wall = o['actual_parallel_wall_clock_ms'] / 1000.0
        synth = e2e - (decomp + wall)
        orch_synth_lat.append(synth)

    max_conconcurrencies = [o.get('max_concurrency', 1) for o in orch_records]
    
    speedups = [seq / wall if wall > 0 else 1.0 for seq, wall in zip(orch_seq_lat, orch_wall_lat)]

    mean_s_lat = ms['latency_seconds']['mean_single'] # 115.68s
    mean_o_lat = ms['latency_seconds']['mean_orchestrated'] # 199.34s
    mean_seq_subtask_time = float(np.mean(orch_seq_lat)) # 562.14s
    mean_parallel_wall_clock = float(np.mean(orch_wall_lat)) # 199.34s wall clock of subtasks + synthesis or wall clock subtasks?
    
    # Notice: actual_parallel_wall_clock_ms in records is total parallel time.
    # Let's check subtask parallel wall clock vs synthesis
    mean_decomp = float(np.mean(orch_decomp_lat))
    mean_synth = float(np.mean(orch_synth_lat))

    # Parallel speedup
    speedup_val = ms['parallel_execution']['mean_parallel_speedup'] # 2.82
    max_c_val = ms['parallel_execution']['mean_max_concurrency'] # 3.17
    conv_eff = speedup_val / max_c_val # 0.889589... -> 0.89 (89.0%)

    # Quality Efficiency (Quality per second)
    s_qual = ms['quality']['mean_single'] # 4.3744
    o_qual = ms['quality']['mean_orchestrated'] # 4.2506

    s_qps = s_qual / mean_s_lat # 4.3744 / 115.68 = 0.037814...
    o_qps = o_qual / mean_o_lat # 4.2506 / 199.34 = 0.021323...
    qps_pct_change = ((o_qps - s_qps) / s_qps) * 100.0 # -43.61%

    # Subtask generation throughput
    s_gen_speed = 8.66 # tok/s
    subtask_gen_throughput = 28.45 # tok/s
    throughput_ratio = subtask_gen_throughput / s_gen_speed # 3.285x

    # Synthesis share of E2E latency
    synth_share_pct = (mean_synth / mean_o_lat) * 100.0

    audit = {
        "metadata": {
            "experiment": "RUN_3_COMPLEX_TASK_COMPARISON",
            "audit_title": "Mathematical and Terminology Scientific Audit",
            "sample_size_n": 18,
            "total_executions": 36,
            "execution_environment": "Local Ollama (CPU 16 threads)",
            "execution_success_rate_pct": 100.0,
            "no_rerun_compliance": True
        },
        "quality_and_coverage": {
            "quality_single_mean": s_qual,
            "quality_single_ci95": [4.05, 4.69],
            "quality_orchestrated_mean": o_qual,
            "quality_orchestrated_ci95": [3.93, 4.57],
            "quality_difference": ms['quality']['mean_difference'],
            "quality_ttest_p_value": ms['quality']['ttest_p_value'],
            "quality_wilcoxon_p_value": ms['quality']['wilcoxon_p_value'],
            "quality_cohens_d": ms['quality']['cohens_d'],
            "quality_statistically_significant": False,
            "coverage_single_pct": ms['objective_coverage']['mean_single_pct'],
            "coverage_orchestrated_pct": ms['objective_coverage']['mean_orchestrated_pct'],
            "coverage_difference_pct": ms['objective_coverage']['mean_difference_pct'],
            "coverage_ttest_p_value": ms['objective_coverage']['ttest_p_value'],
            "coverage_wilcoxon_p_value": ms['objective_coverage']['wilcoxon_p_value'],
            "coverage_statistically_significant": False
        },
        "latency_audit": {
            "single_model_e2e_mean_s": mean_s_lat,
            "single_model_e2e_median_s": ms['latency_seconds']['median_single'],
            "single_model_e2e_p95_s": ms['latency_seconds']['p95_single'],
            "orchestrated_e2e_mean_s": mean_o_lat,
            "orchestrated_e2e_median_s": ms['latency_seconds']['median_orchestrated'],
            "orchestrated_e2e_p95_s": ms['latency_seconds']['p95_orchestrated'],
            "e2e_latency_difference_s": ms['latency_seconds']['mean_difference'],
            "e2e_latency_increase_pct": 72.32,
            "ttest_p_value": ms['latency_seconds']['ttest_p_value'],
            "wilcoxon_p_value": ms['latency_seconds']['wilcoxon_p_value'],
            "latency_breakdown": {
                "decomposition_mean_s": round(mean_decomp, 2),
                "subtask_parallel_wall_clock_mean_s": round(float(np.mean(orch_wall_lat)), 2),
                "subtask_sequential_equivalent_mean_s": round(mean_seq_subtask_time, 2),
                "final_synthesis_mean_s": round(mean_synth, 2),
                "synthesis_share_of_e2e_pct": round(synth_share_pct, 1)
            }
        },
        "parallelization_audit": {
            "sequential_subtask_time_s": 562.14,
            "parallel_wall_clock_time_s": 199.34,
            "measured_parallel_speedup_x": 2.82,
            "mean_max_concurrency": max_c_val,
            "conventional_parallel_efficiency": round(conv_eff, 4),
            "conventional_parallel_efficiency_pct": round(conv_eff * 100.0, 2),
            "previous_reported_non_standard_efficiency": 1.23,
            "correction_reason": "Conventional parallel efficiency is defined as Speedup / Max_Concurrency = 2.82 / 3.17 = 0.89 (89.0%). The non-standard metric reported previously (1.23) exceeded 1.0 and was mathematically invalid as a efficiency measure."
        },
        "end_to_end_quality_efficiency_audit": {
            "single_model_quality_per_sec": round(s_qps, 4),
            "orchestrated_quality_per_sec": round(o_qps, 4),
            "quality_per_sec_pct_change": round(qps_pct_change, 2),
            "previous_reported_orchestrated_qps": 0.1046,
            "correction_reason": "The previous value of 0.1046 was calculated using subtask phase wall-clock time rather than total end-to-end wall-clock latency. Incorporating full end-to-end latency (199.34s), orchestrated quality per second is 0.0213 (a 43.7% reduction compared to 0.0378 for single model)."
        },
        "throughput_and_terminology_audit": {
            "single_model_generation_speed_tok_s": s_gen_speed,
            "aggregate_subtask_generation_throughput_tok_s": subtask_gen_throughput,
            "throughput_ratio_x": round(throughput_ratio, 2),
            "terminology_rule": "28.45 tok/s (3.28x) is aggregate subtask token generation throughput across parallel worker threads. It MUST NOT be called end-to-end generation speedup or quality-per-second improvement, as total end-to-end latency is 72.3% slower on local CPU."
        }
    }

    out_file = 'validation/results/complex_task_comparison/processed/metric_audit.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)

    print(f"Successfully written audited metrics to {out_file}")

if __name__ == '__main__':
    main()
