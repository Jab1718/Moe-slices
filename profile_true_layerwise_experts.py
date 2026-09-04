"""
Master Layer-Wise True Hidden States MoE Profiler.
Directly hooks the input of Qwen4ExpTextTopKRouter across all 48 layers during
real forward passes on multi-lingual coding benchmarks (Python, Rust, C++, Go, Agent).
Projects true hidden states h_l against upstream 512-expert gate matrices to select
the optimal, non-distorted 160 experts per layer (divisible by 16).
All code, comments, and logging strictly in 100% English.
"""

import os
import gc
import json
import time
import math
import numpy as np
import torch
from typing import Dict, List, Tuple
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Hardware constraints on Server Dev02: CUDA 0, 2, 3 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOCAL_MODEL_DIR = "./qwen3.8_flash_bf16_73gb"
UPSTREAM_REPO_ID = "Qwen/Qwen3.8-Flash-Next"
OUTPUT_MAP_PATH = "./true_layerwise_160exp_map.json"
TARGET_EXPERTS_PER_LAYER = 160  # Divisible by 16: 16 x 10 = 160

# -------------------------------------------------------------------------
# Calibration Dataset: Rich Multi-Lingual Tasks from Sandbox Benchmark Suite
# -------------------------------------------------------------------------
def get_calibration_prompts() -> List[Dict[str, str]]:
    tasks = []
    
    # 1. Python Algorithms & Data Structures
    py_cases = [
        ("Python", "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1.\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1\n"),
        ("Python", "def find_kth_largest(nums: list[int], k: int) -> int:\n    import heapq\n    return heapq.nlargest(k, nums)[-1]\n"),
        ("Python", "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    intervals.sort(key=lambda x: x[0])\n    merged = []\n    for inv in intervals:\n        if not merged or merged[-1][1] < inv[0]: merged.append(inv)\n        else: merged[-1][1] = max(merged[-1][1], inv[1])\n    return merged\n"),
        ("Python", "def longest_common_subsequence(text1: str, text2: str) -> int:\n    m, n = len(text1), len(text2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(m):\n        for j in range(n):\n            if text1[i] == text2[j]: dp[i+1][j+1] = dp[i][j] + 1\n            else: dp[i+1][j+1] = max(dp[i][j], dp[i][j+1])\n    return dp[m][n]\n"),
        ("Python", "def coin_change(coins: list[int], amount: int) -> int:\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for c in coins:\n        for a in range(c, amount + 1):\n            dp[a] = min(dp[a], dp[a - c] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1\n"),
        ("Python", "def rotate_matrix_90(matrix: list[list[int]]) -> None:\n    n = len(matrix)\n    for i in range(n):\n        for j in range(i+1, n):\n            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n    for row in matrix:\n        row.reverse()\n"),
        ("Python", "def length_of_lis(nums: list[int]) -> int:\n    import bisect\n    sub = []\n    for x in nums:\n        idx = bisect.bisect_left(sub, x)\n        if idx == len(sub): sub.append(x)\n        else: sub[idx] = x\n    return len(sub)\n"),
        ("Python", "def word_break(s: str, word_dict: list[str]) -> bool:\n    words = set(word_dict)\n    dp = [False] * (len(s) + 1)\n    dp[0] = True\n    for i in range(1, len(s) + 1):\n        for j in range(i):\n            if dp[j] and s[j:i] in words:\n                dp[i] = True; break\n    return dp[len(s)]\n"),
    ]
    for dom, text in py_cases:
        tasks.append({"domain": dom, "text": text})

    # 2. Rust Systems Programming (Borrow checker, Mutex, MPSC, Traits, Iterators)
    rust_cases = [
        ("Rust", "// Send message through tokio mpsc channel\nuse tokio::sync::mpsc;\npub async fn send_event(tx: mpsc::Sender<String>, msg: String) -> Result<(), mpsc::error::SendError<String>> {\n    tx.send(msg).await\n}\n"),
        ("Rust", "// Rust binary search with Result\npub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Result<usize, usize> {\n    let mut left = 0;\n    let mut right = slice.len();\n    while left < right {\n        let mid = left + (right - left) / 2;\n        match slice[mid].cmp(target) {\n            std::cmp::Ordering::Less => left = mid + 1,\n            std::cmp::Ordering::Equal => return Ok(mid),\n            std::cmp::Ordering::Greater => right = mid,\n        }\n    }\n    Err(left)\n}\n"),
        ("Rust", "use std::sync::{Arc, Mutex};\npub fn thread_safe_increment(counter: Arc<Mutex<i32>>) {\n    let mut guard = counter.lock().unwrap();\n    *guard += 1;\n}\n"),
        ("Rust", "struct Counter { count: usize, max: usize }\nimpl Iterator for Counter {\n    type Item = usize;\n    fn next(&mut self) -> Option<Self::Item> {\n        if self.count < self.max { self.count += 1; Some(self.count) } else { None }\n    }\n}\n"),
        ("Rust", "pub fn reverse_words(s: &str) -> String {\n    s.split_whitespace().rev().collect::<Vec<&str>>().join(\" \")\n}\n"),
        ("Rust", "use std::collections::HashMap;\npub fn count_words(text: &str) -> HashMap<String, usize> {\n    let mut map = HashMap::new();\n    for w in text.split_whitespace() { *map.entry(w.to_string()).or_insert(0) += 1; }\n    map\n}\n"),
        ("Rust", "pub trait Summary { fn summarize(&self) -> String; }\npub struct Article { pub title: String, pub author: String }\nimpl Summary for Article {\n    fn summarize(&self) -> String { format!(\"{} by {}\", self.title, self.author) }\n}\n"),
        ("Rust", "enum Shape { Circle(f64), Rectangle(f64, f64) }\nimpl Shape {\n    pub fn area(&self) -> f64 {\n        match self {\n            Shape::Circle(r) => std::f64::consts::PI * r * r,\n            Shape::Rectangle(w, h) => w * h,\n        }\n    }\n}\n"),
    ]
    for dom, text in rust_cases:
        tasks.append({"domain": dom, "text": text})

    # 3. C++ Systems & Modern Standards (Pointers, Templates, Mutex, Concepts)
    cpp_cases = [
        ("CPP", "#include <queue>\n#include <mutex>\ntemplate<typename T>\nclass ThreadSafeQueue {\n    std::queue<T> q;\n    std::mutex m;\npublic:\n    void push(T val) {\n        std::lock_guard<std::mutex> lock(m);\n        q.push(val);\n    }\n};\n"),
        ("CPP", "#include <vector>\nint binary_search(const std::vector<int>& arr, int target) {\n    int left = 0, right = static_cast<int>(arr.size()) - 1;\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1\n}\n"),
        ("CPP", "#include <memory>\n#include <string>\nstruct User { std::string name; int age; };\nstd::unique_ptr<User> make_user(const std::string& name, int age) {\n    return std::make_unique<User>(User{name, age});\n}\n"),
        ("CPP", "struct Node {\n    int key, val;\n    Node *prev, *next;\n    Node(int k, int v) : key(k), val(v), prev(nullptr), next(nullptr) {}\n};\nvoid link_nodes(Node* a, Node* b) {\n    a->next = b;\n    b->prev = a;\n}\n"),
        ("CPP", "#include <concepts>\ntemplate<std::integral T>\nT add(T a, T b) {\n    return a + b;\n}\n"),
        ("CPP", "#include <atomic>\nstruct Counter {\n    std::atomic<int> val{0};\n    void increment() { val.fetch_add(1, std::memory_order_relaxed); }\n};\n"),
        ("CPP", "#include <algorithm>\n#include <vector>\nvoid sort_descending(std::vector<int>& nums) {\n    std::sort(nums.begin(), nums.end(), std::greater<int>());\n}\n"),
        ("CPP", "#include <future>\nstd::future<int> async_compute(int a, int b) {\n    return std::async(std::launch::async, [a, b]() { return a * b + 42; });\n}\n"),
    ]
    for dom, text in cpp_cases:
        tasks.append({"domain": dom, "text": text})

    # 4. Go Concurrent & Backend Systems
    go_cases = [
        ("Go", "package main\nfunc worker(id int, jobs <-chan int, results chan<- int) {\n    for j := range jobs {\n        results <- j * 2\n    }\n}\n"),
        ("Go", "package main\nimport \"sync\"\ntype SafeMap struct {\n    mu sync.RWMutex\n    m map[string]int\n}\nfunc (s *SafeMap) Get(key string) (int, bool) {\n    s.mu.RLock()\n    defer s.mu.RUnlock()\n    val, ok := s.m[key]\n    return val, ok\n}\n"),
        ("Go", "package main\nfunc BinarySearch(arr []int, target int) int {\n    left, right := 0, len(arr)-1\n    for left <= right {\n        mid := left + (right-left)/2\n        if arr[mid] == target { return mid }\n        if arr[mid] < target { left = mid + 1 } else { right = mid - 1 }\n    }\n    return -1\n}\n"),
        ("Go", "package main\ntype User struct {\n    Name string `json:\"name\"`\n    Age int `json:\"age\"`\n}\n"),
        ("Go", "package main\nimport \"net/http\"\nfunc HealthCheckHandler(w http.ResponseWriter, r *http.Request) {\n    w.WriteHeader(http.StatusOK)\n    w.Write([]byte(\"OK\"))\n}\n"),
    ]
    for dom, text in go_cases:
        tasks.append({"domain": dom, "text": text})

    # 5. Coding Agent & Tool Calling Trajectories
    agent_cases = [
        ("Agent", "<|im_start|>system\nCall grep_search(Query: str, SearchPath: str) in format: {\"tool\": \"grep_search\", \"parameters\": {\"Query\": \"...\", \"SearchPath\": \"...\"}}<|im_end|>\n<|im_start|>user\nFind 'def binary_search' in /src.<|im_end|>\n<|im_start|>assistant\n{\"tool\": \"grep_search\", \"parameters\": {\"Query\": \"def binary_search\", \"SearchPath\": \"/src\"}}<|im_end|>\n"),
        ("Agent", "<|im_start|>system\nCall read_file(path: str, start_line: int, end_line: int) in format: {\"tool\": \"read_file\", \"parameters\": {\"path\": \"...\", \"start_line\": 1, \"end_line\": 10}}<|im_end|>\n<|im_start|>user\nRead lines 10 to 50 of /workspace/main.py.<|im_end|>\n<|im_start|>assistant\n{\"tool\": \"read_file\", \"parameters\": {\"path\": \"/workspace/main.py\", \"start_line\": 10, \"end_line\": 50}}<|im_end|>\n"),
        ("Agent", "<|im_start|>system\nCall write_to_file(path: str, content: str) in format: {\"tool\": \"write_to_file\", \"parameters\": {\"path\": \"...\", \"content\": \"...\"}}<|im_end|>\n<|im_start|>user\nCreate file /app/test.py with content 'print(1)'.<|im_end|>\n<|im_start|>assistant\n{\"tool\": \"write_to_file\", \"parameters\": {\"path\": \"/app/test.py\", \"content\": \"print(1)\"}}<|im_end|>\n"),
        ("Agent", "<|im_start|>system\nCall run_command(command: str) in format: {\"tool\": \"run_command\", \"parameters\": {\"command\": \"...\"}}<|im_end|>\n<|im_start|>user\nRun pytest on tests/ directory.<|im_end|>\n<|im_start|>assistant\n{\"tool\": \"run_command\", \"parameters\": {\"command\": \"pytest tests/\"}}<|im_end|>\n"),
    ]
    for dom, text in agent_cases:
        tasks.append({"domain": dom, "text": text})

    return tasks


def run_true_layerwise_profiling():
    print("=" * 90)
    print(" 🎯 LAYER-WISE TRUE HIDDEN STATES MOE EXPERT PROFILER")
    print(f"    • Local Subnet Source     : {LOCAL_MODEL_DIR}")
    print(f"    • Upstream Repository     : {UPSTREAM_REPO_ID}")
    print(f"    • Target Experts / Layer  : {TARGET_EXPERTS_PER_LAYER} / 512 (Divisible by 16)")
    print(f"    • Output Selection Map    : {OUTPUT_MAP_PATH}")
    print("=" * 90)

    # 1. Resolve Upstream 512-Expert Gate Weights from Hugging Face Cache
    print("\n[Step 1/5] Loading 48 Upstream Gate Weights (512 experts each) from cache...")
    index_file = hf_hub_download(repo_id=UPSTREAM_REPO_ID, filename="model.safetensors.index.json")
    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})

    upstream_gates: Dict[int, torch.Tensor] = {}
    gate_locations = {}
    for l_idx in range(48):
        k1 = f"model.language_model.layers.{l_idx}.mlp.gate.weight"
        k2 = f"model.layers.{l_idx}.mlp.gate.weight"
        key = k1 if k1 in weight_map else k2
        gate_locations[l_idx] = (key, weight_map[key])

    shard_to_layers = {}
    for l_idx, (k, s) in gate_locations.items():
        shard_to_layers.setdefault(s, []).append(l_idx)

    for shard_name, layers in shard_to_layers.items():
        shard_path = hf_hub_download(repo_id=UPSTREAM_REPO_ID, filename=shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for l_idx in layers:
                k, _ = gate_locations[l_idx]
                W_g = f.get_tensor(k).to(torch.float32)  # Shape: (512, 2560)
                upstream_gates[l_idx] = W_g
    print(f"[✓] Loaded all 48 upstream gate tensors into CPU RAM: {len(upstream_gates)} layers.")

    # 2. Load Local Backbone Model Across 3 GPUs
    print("\n[Step 2/5] Loading Model across 3 GPUs for True Hidden State extraction...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
    
    device_map = {
        "model.embed_tokens": "cpu",
        "model.rotary_emb": "cuda:0",
        "model.hyper_connection_mixer": "cuda:2",
        "model.norm": "cuda:2",
        "lm_head": "cuda:2"
    }
    for i in range(14):
        device_map[f"model.layers.{i}"] = "cuda:0"
    for i in range(14, 31):
        device_map[f"model.layers.{i}"] = "cuda:1"
    for i in range(31, 48):
        device_map[f"model.layers.{i}"] = "cuda:2"

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        dtype=torch.bfloat16,
        device_map=device_map,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )
    model.eval()
    print(f"[✓] Model loaded on 3 GPUs (Dynamic 14-17-17 + CPU Embed) in {time.perf_counter() - t0:.2f}s!")

    # 3. Setup On-The-Fly Profiling Accumulators
    print("\n[Step 3/5] Registering Pre-Forward Hooks to capture true hidden states h_l...")
    
    # Per-layer statistics:
    # freq_counts[layer]: (512,) int - number of times expert was in Top-10
    # sum_scores[layer]: (512,) float - sum of sigmoid activation scores
    # domain_freq[layer][domain]: (512,) int
    freq_counts = {l: np.zeros(512, dtype=np.int64) for l in range(48)}
    sum_scores = {l: np.zeros(512, dtype=np.float64) for l in range(48)}
    domain_freq = {l: {} for l in range(48)}
    token_counts = {l: 0 for l in range(48)}

    current_domain = ["Python"]  # mutable holder for current domain in hook

    # Hook function registered on layer.mlp.gate
    def make_gate_hook(layer_idx: int):
        W_gate = upstream_gates[layer_idx]  # Shape: (512, 2560) on CPU

        def hook(module, inp):
            # inp is a tuple: (hidden_states,)
            h = inp[0].detach()  # Shape: [batch, seq_len, 2560]
            # Flatten to (N, 2560) and move to CPU float32
            h_flat = h.view(-1, 2560).to(torch.float32).cpu()
            n_tokens = h_flat.shape[0]
            token_counts[layer_idx] += n_tokens

            # Compute True Layer-Wise Logits against upstream 512-expert gate:
            logits = torch.matmul(h_flat, W_gate.t())  # (N, 512)
            scores = torch.sigmoid(logits)             # (N, 512)
            
            # Top-10 experts per token
            _, top10_indices = torch.topk(scores, k=10, dim=-1)  # (N, 10)
            
            # Update accumulators
            top10_np = top10_indices.numpy()
            scores_np = scores.numpy()
            
            # Fast bincount
            flat_indices = top10_np.ravel()
            bincount = np.bincount(flat_indices, minlength=512)
            freq_counts[layer_idx] += bincount
            sum_scores[layer_idx] += np.sum(scores_np, axis=0)

            dom = current_domain[0]
            if dom not in domain_freq[layer_idx]:
                domain_freq[layer_idx][dom] = np.zeros(512, dtype=np.int64)
            domain_freq[layer_idx][dom] += bincount

        return hook

    hook_handles = []
    for l_idx in range(48):
        handle = model.model.layers[l_idx].mlp.gate.register_forward_pre_hook(make_gate_hook(l_idx))
        hook_handles.append(handle)
    print(f"[✓] Successfully attached 48 hooks to mlp.gate across all layers.")

    # 4. Ingest Calibration Benchmark Corpus
    print("\n[Step 4/5] Executing Forward Passes across Multi-Lingual Calibration Tasks...")
    corpus = get_calibration_prompts()
    print(f"    • Total calibration tasks : {len(corpus)}")
    
    t_fwd_start = time.perf_counter()
    with torch.no_grad():
        for item in corpus:
            dom = item["domain"]
            text = item["text"]
            current_domain[0] = dom
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            _ = model(**inputs)

    # Remove hooks
    for h in hook_handles:
        h.remove()
    print(f"[✓] Forward profiling finished in {time.perf_counter() - t_fwd_start:.2f}s!")
    print(f"    • Total tokens evaluated per layer: ~{token_counts[0]:,} tokens.")

    # Free model from GPU memory to allow slicing / evaluation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[✓] Model memory released from 3 GPUs.")

    # 5. Synthesize True Layer-Wise 160-Expert Selection Map
    print("\n[Step 5/5] Synthesizing Selection Map (160 Experts / Layer)...")
    final_selection_map: Dict[int, List[int]] = {}
    domains = ["Python", "Rust", "CPP", "Go", "Agent"]

    total_pinned = 0
    for l_idx in range(48):
        T = token_counts[l_idx]
        freq = freq_counts[l_idx] / max(1, T)        # (512,)
        avg_score = sum_scores[l_idx] / max(1, T)    # (512,)

        # Compute Domain Specificity (TF-IDF style)
        # Exclusivity: fraction of domain's tokens that picked expert e
        domain_spec = np.zeros(512, dtype=np.float64)
        for dom in domains:
            if dom in domain_freq[l_idx]:
                df = domain_freq[l_idx][dom]
                dom_total = np.sum(df) + 1e-8
                ratio = df / dom_total
                # Exclusivity relative to global frequency
                exclusivity = ratio / (freq + 1e-6)
                domain_spec = np.maximum(domain_spec, exclusivity)

        # Combined Ranking Score: 50% Activation Score + 30% Frequency + 20% Specificity
        combined_score = 0.50 * avg_score + 0.30 * freq + 0.20 * (domain_spec / (np.max(domain_spec) + 1e-6))

        selected = set()

        # Step 5a: Critical Domain Pinning (Top-12 specialists per domain to guarantee zero amnesia)
        for dom in domains:
            if dom in domain_freq[l_idx]:
                df = domain_freq[l_idx][dom]
                top_dom = np.argsort(df)[::-1][:12]
                for e in top_dom:
                    selected.add(int(e))

        total_pinned += len(selected)

        # Step 5b: Fill remaining up to 160 by combined score
        ranked_experts = np.argsort(combined_score)[::-1]
        for e in ranked_experts:
            if len(selected) >= TARGET_EXPERTS_PER_LAYER:
                break
            selected.add(int(e))

        sorted_list = sorted(list(selected))
        assert len(sorted_list) == TARGET_EXPERTS_PER_LAYER, f"Layer {l_idx} has {len(sorted_list)} experts != 160!"
        final_selection_map[l_idx] = sorted_list

    print(f"[✓] Successfully selected {TARGET_EXPERTS_PER_LAYER} experts across all 48 layers.")
    print(f"    • Guaranteed Domain-Pinned Experts: {total_pinned} assignments across layers.")

    with open(OUTPUT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(final_selection_map, f, indent=2)
    print(f"[✓] Selection map saved to: {OUTPUT_MAP_PATH}")

    # Summary statistics
    all_unique = set()
    for l_idx, exps in final_selection_map.items():
        all_unique.update(exps)
    print(f"    • Global Unique Experts Selected across 48 layers: {len(all_unique)} / 512")
    print("=" * 90)
    print(" ✅ TRUE LAYER-WISE PROFILING COMPLETE! READY FOR STREAMING RE-SLICE.")
    print("=" * 90)


if __name__ == "__main__":
    run_true_layerwise_profiling()
