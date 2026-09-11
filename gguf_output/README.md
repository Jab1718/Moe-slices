---
language:
- en
- vi
- zh
license: apache-2.0
tags:
- gguf
- llama-cpp
- ollama
- moe
- qwen
- code
- coding-agent
- quantization
base_model: Jab1718/qwen3.8-flash-coder-85gb-bf16
pipeline_tag: text-generation
---

# ⚡ Qwen3.8-Flash-Coder-26GB-GGUF (160 Experts Hardware-Aligned Coding Subnet)

[![GitHub Toolkit](https://img.shields.io/badge/GitHub-Jab1718%2FMoe--slices-blue?logo=github)](https://github.com/Jab1718/Moe-slices)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![BF16 Base](https://img.shields.io/badge/BF16%20Base-Jab1718%2Fqwen3.8--flash--coder--85gb--bf16-orange)](https://huggingface.co/Jab1718/qwen3.8-flash-coder-85gb-bf16)
[![Selective INT8](https://img.shields.io/badge/INT8%20Subnet-Jab1718%2Fqwen3.8--flash--coder--44gb--selective--int8-purple)](https://huggingface.co/Jab1718/qwen3.8-flash-coder-44gb-selective-int8)

This repository provides **GGUF** quantized formats of [**`Jab1718/qwen3.8-flash-coder-85gb-bf16`**](https://huggingface.co/Jab1718/qwen3.8-flash-coder-85gb-bf16), a 160-expert coding subnet sliced from `Qwen3.8-Flash-Next` (335GB) using the [**`moe-slice`**](https://github.com/Jab1718/Moe-slices) toolkit.

---

## 📦 Provided GGUF Files

| File Name | Quantization | Size on Disk | Target Hardware | Recommended Use Case |
| :--- | :---: | :---: | :--- | :--- |
| **`qwen3.8-flash-coder-26gb-q4_k_m.gguf`** | **Q4_K_M** | **26.43 GB (27 GB)** | **1x 24GB GPU** (RTX 3090/4090), RTX 5000 Ada (32GB), or 32GB Mac | **🔥 92% VRAM reduction from 335GB base! Run locally on single consumer GPU!** |
| **`qwen3.8-flash-coder-43gb-q8_0.gguf`** | **Q8_0** | **42.25 GB (43 GB)** | **2x 24GB GPUs** / 1x 48GB GPU / 64GB+ RAM Mac | **Maximum reasoning fidelity (~99.5% BF16)** |
| **`Modelfile`** | Text config | <1 KB | Ollama | Quick import and deployment in Ollama |

---

## 🚀 Quick Start with Ollama

### Step 1: Download the GGUF file and Modelfile
```bash
huggingface-cli download Jab1718/qwen3.8-flash-coder-26gb-gguf qwen3.8-flash-coder-26gb-q4_k_m.gguf --local-dir .
huggingface-cli download Jab1718/qwen3.8-flash-coder-26gb-gguf Modelfile --local-dir .
```

### Step 2: Create and run model in Ollama
```bash
ollama create qwen3.8-flash-coder -f Modelfile
ollama run qwen3.8-flash-coder "Implement a thread-safe LRU Cache with TTL expiration in Rust."
```

---

## 🍏 Apple Silicon & Mac Setup Notes (Metal Acceleration)

> [!IMPORTANT]
> **Runtime Compatibility Notice for macOS Users:**
> - **Ollama & Legacy Runtimes:** Bundled Ollama versions (e.g., v0.32.5 or earlier) and older DS4 binaries hardcode `expert_count=512` and will reject this file (`expected expert_count=512 … got 160`).
> - **Recommended Runtime:** Build and run using the latest **`llama.cpp master`** with Metal support (which includes native compatibility PRs for arbitrary MoE expert counts like `qwen4exp`).

### ⚡ Verified Apple Silicon Performance Profile (Tested on Apple M5 Pro, 64 GB Unified Memory, macOS 26)
* **Extraordinary Memory Efficiency:** Loads at full **64K context (`-c 65536`) using only ~33 GB wired RAM** (~46% RAM free) — massive headroom for concurrent developer workflows on 64 GB Macs!
* **Blazing Fast Decode:** **~32 tokens/second** sustained decode throughput.
* **Near-Instant Initialization:** Cold load in **~10 seconds** via memory-mapped I/O.

### Recommended `llama-server` Launch Command:
```bash
./llama-server \
  -m ./qwen3.8-flash-coder-26gb-q4_k_m.gguf \
  -c 65536 \
  -ngl 999 \
  --host 0.0.0.0 \
  --port 8080
```

---

## 💡 Prompting Guidelines & Best Practices

### 1. Code Review & Bug-Finding Prompts (Preventing Empty Content)
* **The Symptom:** In analytical/code-auditing tasks (e.g., *"Find all bugs in this function"*), certain frontends/chat templates may capture reasoning inside `<think>...</think>` but route near-empty text (0–2 chars) into the final `content` field upon encountering `finish_reason=stop`.
* **Root Cause:** Chat template stop-token routing or missing explicit conclusion markers outside the thought block.
* **Solution (Explicit Output Routing):** Instruct the model explicitly to provide its structured audit report **outside** the thought tags:

```markdown
# Recommended Prompt Pattern:
Review the following code for concurrency bugs and memory leaks.
Think thoroughly through edge cases inside <think>...</think>.
CRITICAL: After closing the </think> tag, you MUST provide your final structured bug report (specifying Bug Name, Line Number, Severity, and Recommended Fix) directly in your response content.
```

### 2. Surgical Code Refactoring (Preventing Over-Rewrites)
* To prevent the model from rewriting entire algorithms from scratch (which can introduce edge-case regressions like off-by-one errors):
* **Best Practice:** Prompt the model for a **minimal diff** or **surgical patch**:
```markdown
# Recommended Refactoring Prompt:
Refactor the following function to fix the infinite loop bug.
Provide ONLY a minimal surgical unified diff patch (+/- lines) without rewriting unaffected logic.
```

### 3. Domain Focus & General Math Trade-off
* **Note on Math/GSM8K:** This 160-expert subnet was intentionally extracted by prioritizing Systems Programming, Algorithmic DP, and Tool-Calling trajectories. General word-problem arithmetic (GSM8K) was pruned by design to give maximum capacity to multi-language software development.

---

## 📊 Benchmark Capabilities

* **Rust (Systems):** **10/10 (100.0%)** Pass@1
* **C++20 (Modern Systems):** **10/10 (100.0%)** Pass@1
* **TypeScript (Fullstack):** **4/5 (80.0%)** Pass@1
* **Coding Agent (Diff/FIM/Bugfix):** **16/20 (80.0%)** Pass@1
* **Python Algorithms:** **39/50 (78.0%)** Pass@1

---

## 📜 Citation

```bibtex
@software{thainq2026moeslice_gguf,
  author = {Thai Nguyen},
  title = {Qwen3.8-Flash-Coder-GGUF: Hardware-Aligned Coding Subnet for llama.cpp and Ollama},
  url = {https://github.com/Jab1718/Moe-slices},
  year = {2026}
}
```
