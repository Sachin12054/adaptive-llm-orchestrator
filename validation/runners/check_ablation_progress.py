import json
import os

def main():
    res_file = 'validation/results/complex_task_synthesis_optimization/raw/synthesis_ablation_results.jsonl'
    if not os.path.exists(res_file):
        print("No results recorded yet.")
        return

    with open(res_file, 'r', encoding='utf-8') as f:
        recs = [json.loads(l) for l in f if l.strip()]

    print(f"Total trial records completed so far: {len(recs)}")
    pids = list(set(r["prompt_id"] for r in recs))
    print(f"Prompts completed ({len(pids)}/18): {pids}")

    for r in recs:
        print(f" - Prompt: {r['prompt_id']} | Strategy: {r['strategy_id']} | Synth Latency: {r['synthesis_latency_ms']/1000.0:.2f}s | Total E2E: {r['total_e2e_latency_ms']/1000.0:.2f}s | Quality: {r['quality_score_0_to_5']} | Coverage: {r['objective_coverage_pct']}%")

if __name__ == '__main__':
    main()
