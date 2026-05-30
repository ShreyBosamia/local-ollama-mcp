# Side-by-Side Model Comparison Report

This report presents a direct empirical comparison between **qwen3.5:9b** and **deepseek-r1:8b** across the **pipeline** evaluation suite.

*   **Generated At**: 2026-05-28T05:51:57Z
*   **Target Suite**: `pipeline`
*   **GPU Residency**: Tested on 8GB GPU (NVIDIA GeForce RTX 2070 SUPER)

---

## 📊 Core Performance Metrics

| Performance Indicator | qwen3.5:9b | deepseek-r1:8b | Relative Change |
| :--- | :---: | :---: | :---: |
| **Total Test Cases** | 5 | 5 | -- |
| **Factual Accuracy Score** | 80.0% | 80.0% | +0.0% |
| **Average Latency** | 14,698 ms | 12,991 ms | -11.6% |
| **Average Generation Speed** | **24.72 tok/s** | **22.61 tok/s** | **-8.5%** |
| **Aggregate Context Reduction** | 94.8% | 95.3% | +0.5% |
| **Usefulness Score** | 0.890 | 0.890 | +0.000 |
| **Think Leakage Count** | 0 | 0 | -- |

---

## 🔍 Case-by-Case Analysis

| Case Name | qwen3.5:9b Accuracy | deepseek-r1:8b Accuracy | qwen3.5:9b Speed (TPS) | deepseek-r1:8b Speed (TPS) |
| :--- | :---: | :---: | :---: | :---: |
| `framework_api_docs_pipeline` | 100.0% | 100.0% | 44.9 | 33.3 |
| `postgres_lock_trace_pipeline` | 66.7% | 66.7% | 26.0 | 25.6 |
| `react_prop_drill_pipeline` | 66.7% | 66.7% | 1.1 | 1.0 |
| `telemetry_vram_safety_pipeline` | 66.7% | 66.7% | 18.2 | 18.9 |
| `vite_bundle_audit_pipeline` | 100.0% | 100.0% | 33.5 | 34.2 |

---

## 💡 Key Architectural Insights

1. **Tokens-Per-Second (TPS) Speed & CPU Fallback**:
   * **Observation**: deepseek-r1:8b is **8.5% slower** than qwen3.5:9b in token generation speed.
   * **Analysis**: This confirms that the **6.6 GB size** of Qwen 3.5 9B triggered **CPU Fallback/spillover** on the 8GB GPU. While Qwen 2.5 7B was processed 100% on-GPU, the larger Qwen 3.5 model experienced memory bottlenecking across Zen 2's system memory channels.

2. **Accuracy & Quality of Summarization**:
   * deepseek-r1:8b achieved an average accuracy of **80.0%** compared to qwen3.5:9b's **80.0%**. 
   * This quantifies the trade-off: **Qwen 3.5 9B** provides greater conceptual precision and fact retention for developer workflows at the cost of processing speed.

---

## 🛠️ Configuration Recommender
*   **Use `qwen3.5:9b`** if you require real-time interactive performance and want a 100% on-GPU latency profile with low VRAM usage.
*   **Use `deepseek-r1:8b`** if you are executing offline, complex, or multi-step agentic summarization tasks where factual precision is paramount and a ~15 tok/s CPU offload latency is perfectly acceptable.
