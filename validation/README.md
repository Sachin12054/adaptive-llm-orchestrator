# Scientific Validation & Benchmarking Framework

**Project**: Adaptive Multi-LLM Orchestration Using Contextual Bandits  
**Dataset Version**: `v1.0.0` (Exactly **113 Prompts** across **15 Categories**)  
**Production Authority**: `BaselineAdaptivePolicy` (`production_override = false`)  
**RL Evaluator**: `RLContextualBanditPolicy` (Shadow Mode Only)  

---

## Architectural Purpose

This validation suite provides a rigorous, reproducible, scientific benchmark evaluating **BaselineAdaptivePolicy** and **RLContextualBanditPolicy** against 6 baseline routing strategies across Quality, Latency, Cost, Reliability, Resource Efficiency, and Statistical Significance.

### Non-Negotiable Invariants
1. **`BaselineAdaptivePolicy` remains the sole production authority.**
2. **`RLContextualBanditPolicy` remains SHADOW-ONLY.**
3. **`production_override = false` is enforced across all runners.**
4. **`BAAI/bge-m3` (Action 7) is masked from text generation.**

---

## Directory Structure

```text
validation/
├── README.md                           # Framework overview & sitemap
├── config/
│   ├── benchmark_config.json           # Seed=42, 113 prompts, safety invariants
│   ├── evaluation_config.json          # Quality rubric (0-5 scale) & provider pricing
│   └── strategy_config.json            # Model parameters for Strategies A–H
├── datasets/
│   ├── benchmark_dataset.jsonl         # 113 prompts across 15 categories (v1.0.0)
│   ├── benchmark_metadata.json         # Category count metadata
│   └── dataset_summary.md              # Category distribution documentation
├── strategies/
│   ├── fixed_model_policy.py           # Strategies A, B, C (Fixed Gemma, Qwen, DeepSeek)
│   ├── random_policy.py                # Strategy D (Reproducible random routing, seed=42)
│   ├── round_robin_policy.py           # Strategy E (Round-robin model rotation)
│   ├── capability_heuristic_policy.py  # Strategy F (Rule-based capability heuristic)
│   ├── baseline_adaptive_policy_adapter.py # Strategy G (Production authority adapter)
│   └── rl_shadow_policy_adapter.py     # Strategy H (Offline RL Shadow evaluator)
├── metrics/
│   ├── quality_metrics.py              # 0-5 rubric scoring & domain consistency
│   ├── latency_metrics.py              # Preprocessing, routing, E2E, P50/P95, speedup
│   ├── cost_metrics.py                 # Input/output token costs
│   ├── reliability_metrics.py          # Success, timeout, failover rates
│   ├── resource_metrics.py             # CPU, RAM, GPU, VRAM telemetry
│   └── statistical_metrics.py          # Mean, 95% CIs, IPS, SNIPS, ESS, bootstrap CIs, p-values
├── runners/
│   ├── run_all_validations.py          # Master CLI runner (--resume, --dry-run, --fast)
│   ├── run_strategy_comparison.py      # Live strategy runner (A–G)
│   ├── run_latency_benchmark.py        # Latency breakdown runner
│   ├── run_quality_benchmark.py        # Quality rubric evaluation runner
│   ├── run_rl_shadow_evaluation.py     # Offline RL IPS/SNIPS/ESS evaluator
│   └── run_statistical_analysis.py     # Hypothesis testing & bootstrap CI runner
├── results/
│   ├── raw/                            # Telemetry log outputs
│   ├── processed/                      # Per-strategy summary JSON files
│   ├── statistical/                    # Bootstrap CIs, ESS, IPS policy values
│   ├── figures/                        # Generated PNG plots (16 research charts)
│   ├── final_validation_summary.json   # Comprehensive 8-strategy summary table
│   └── experiment_manifest.json        # Execution manifest & code versions
└── reports/                            # 14 Auto-Generated Markdown Research Reports (01–14)
```

---

## Quick Start Command

To execute the complete automated scientific validation workflow:

```bash
python validation/runners/run_all_validations.py
```

### Options:
- `--dry-run`: Validate configurations and datasets without invoking external APIs.
- `--resume`: Resume interrupted strategy execution from existing raw telemetry logs.
- `--fast`: Execute a fast sub-sample validation across all 15 task categories.
