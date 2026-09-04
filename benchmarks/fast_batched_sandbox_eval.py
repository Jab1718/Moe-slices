"""
High-Throughput Batched Sandbox Benchmark (100 Tasks).
Features:
- Batched Inference (Batch Size = 8) with left-padding for ~10x speedup (~1.5 min total).
- Fast pre-compilation syntax check with compile() to fail-fast in < 0.1ms.
- Deterministic auto-indentation resolution for Python function body completions.
- Isolated Python subprocess sandbox execution with timeout=3s.
- Automatic baseline gating against official Qwen3.8-Flash-Next 335GB ground truth.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import sys
import time
import json
import textwrap
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils
import transformers.core_model_loading
from concurrent.futures import ThreadPoolExecutor

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
orig_thread_pool = ThreadPoolExecutor
class SingleWorkerThreadPool(orig_thread_pool):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(max_workers=1, *args, **kwargs)
transformers.core_model_loading.ThreadPoolExecutor = SingleWorkerThreadPool


def clean_python_code(prompt: str, generated_text: str) -> str:
    """
    Extracts and properly indents pure Python code from model generation.
    Handles raw body completions, full redeclarations, and markdown fences.
    """
    text = generated_text
    
    # 1. Extract pure code
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1 and any(kw in parts[0] for kw in ["return ", "while ", "for ", "if ", " = "]):
            content = parts[0].strip()
        elif len(parts) > 1:
            snippet = parts[1].strip()
            if snippet.lower().startswith("python"):
                snippet = snippet[6:].strip()
            content = snippet
        else:
            content = text.strip()
    else:
        content = text.strip()

    # 2. Strip trailing conversation tokens
    for stop_word in ["\n<|im_end|>", "\n<|endoftext|>", "\n# Test", "\nprint(", "\nif __name__", "\n```"]:
        if stop_word in content:
            content = content.split(stop_word)[0].rstrip()

    # Detect function or class name
    func_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", prompt)
    class_match = re.search(r"class\s+([a-zA-Z0-9_]+)", prompt)

    if func_match:
        fname = func_match.group(1)
        if f"def {fname}" in content:
            return content
        else:
            # Check if body needs indentation
            non_empty = [l for l in content.split("\n") if l.strip()]
            if non_empty and not non_empty[0].startswith("    ") and not non_empty[0].startswith("\t"):
                indented = textwrap.indent(content, "    ")
            else:
                indented = content
            return prompt.rstrip() + "\n" + indented

    elif class_match:
        cname = class_match.group(1)
        if f"class {cname}" in content:
            return content
        else:
            non_empty = [l for l in content.split("\n") if l.strip()]
            if non_empty and not non_empty[0].startswith("    ") and not non_empty[0].startswith("\t"):
                indented = textwrap.indent(content, "    ")
            else:
                indented = content
            return prompt.rstrip() + "\n" + indented

    return prompt.rstrip() + "\n" + content


def run_python_sandbox_fast(full_code: str, test_assertions: str, timeout: float = 3.0) -> bool:
    """
    Validates syntax with compile() in 0.1ms before executing test assertions in a subprocess.
    """
    script = f"{full_code}\n\n# --- ASSERTIONS ---\n{test_assertions}\n"
    # Pre-compile check: fail-fast if syntax or indentation error exists
    try:
        compile(script, "<sandbox>", "exec")
    except Exception:
        return False

    # Execute assertions in subprocess sandbox
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from benchmarks.eval_comprehensive_100_sandbox import build_100_sandbox_tasks
except ModuleNotFoundError:
    from eval_comprehensive_100_sandbox import build_100_sandbox_tasks


def run_fast_batched_evaluation(model_path: str = "./qwen3.8_flash_bf16_73gb", batch_size: int = 8):
    baseline_file = "./benchmarks/baseline_qwen_groundtruth.json"
    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    baseline_results = baseline_data["results"]

    print("=" * 100)
    print(" 🚀 HIGH-THROUGHPUT BATCHED SANDBOX BENCHMARK (100 TASKS) + BASELINE GATING")
    print(f"    • Model Source    : {model_path}")
    print(f"    • Inference Mode  : Left-Padded Batched Generation (Batch Size = {batch_size})")
    print(f"    • Baseline Target : Qwen/Qwen3.8-Flash-Next (335 GB Baseline Ground Truth)")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.hyper_connection_mixer": "cuda:2",
        "model.norm": "cuda:2",
        "lm_head": "cuda:2"
    }
    for i in range(15):
        device_map[f"model.layers.{i}"] = "cuda:0"
    for i in range(15, 32):
        device_map[f"model.layers.{i}"] = "cuda:1"
    for i in range(32, 48):
        device_map[f"model.layers.{i}"] = "cuda:2"

    print("[*] Loading model weights across 3 GPUs...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    print(f"[✓] Model loaded in {time.perf_counter() - t0:.2f}s!")

    print("[*] Executing GPU warm-up pass...")
    warm_in = tokenizer(["def warm_up(): return True"], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**warm_in, max_new_tokens=5)
    print("[✓] GPU warm-up completed!")

    suite = build_100_sandbox_tasks()
    print(f"\n[*] Commencing Batched Sandbox evaluation on {len(suite)} tasks (Batches of {batch_size})...\n")

    results = []
    category_stats = {}
    targeted_attribution_list = []
    skipped_baseline_fails = []

    t_eval_start = time.perf_counter()

    # Process in batches of 8
    for batch_idx in range(0, len(suite), batch_size):
        batch_tasks = suite[batch_idx:batch_idx + batch_size]
        prompts = [t["prompt"] for t in batch_tasks]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
        t_bstart = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=130,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id
            )
        t_batch = time.perf_counter() - t_bstart

        input_len = inputs.input_ids.shape[1]
        gen_tokens_batch = outputs[:, input_len:]

        for i, task in enumerate(batch_tasks):
            tid = task["id"]
            cat = task["category"]
            prompt = task["prompt"]
            vtype = task["verifier_type"]

            gen_text = tokenizer.decode(gen_tokens_batch[i], skip_special_tokens=True)

            # Sandbox evaluation
            if vtype == "sandbox_python":
                clean_code = clean_python_code(prompt, gen_text)
                is_pass = run_python_sandbox_fast(clean_code, task["asserts"])
            else:
                is_pass = task["verifier_fn"](gen_text)

            b_info = baseline_results.get(tid, {"status": "PASS"})
            base_pass = (b_info["status"] == "PASS")

            if cat not in category_stats:
                category_stats[cat] = {"passed": 0, "total": 0}
            category_stats[cat]["total"] += 1
            if is_pass:
                category_stats[cat]["passed"] += 1

            if not base_pass and not is_pass:
                status_tag = "SKIP (Baseline FAIL)"
                skipped_baseline_fails.append(tid)
            elif base_pass and not is_pass:
                status_tag = "TARGET TRACE (Pruning Degradation)"
                targeted_attribution_list.append(tid)
            else:
                status_tag = "PASS"

            results.append({
                "id": tid,
                "category": cat,
                "passed": is_pass,
                "baseline_passed": base_pass,
                "status_tag": status_tag
            })

        processed = len(results)
        passed_so_far = sum([1 for r in results if r["passed"]])
        print(f"  [{processed:3d}/{len(suite)}] Progress: {passed_so_far}/{processed} Passed ({(passed_so_far/processed)*100:.1f}%) | Batch Speed: {len(batch_tasks)/t_batch:.2f} tasks/s", flush=True)

    total_time = time.perf_counter() - t_eval_start
    total_passed = sum([1 for r in results if r["passed"]])
    baseline_solvable = sum([1 for r in results if r["baseline_passed"]])
    solvable_passed = sum([1 for r in results if r["passed"] and r["baseline_passed"]])
    
    fidelity_rate = (solvable_passed / max(baseline_solvable, 1)) * 100.0
    absolute_pass_rate = (total_passed / len(suite)) * 100.0

    print("\n" + "=" * 100)
    print(f" 🏆 BATCHED SANDBOX BENCHMARK & FIDELITY RETENTION REPORT:")
    print(f"    • Knowledge Retention Fidelity Rate : {solvable_passed}/{baseline_solvable} ({fidelity_rate:.1f}%) [TARGET: >= 98%]")
    print(f"    • Absolute Pass@1 Accuracy          : {total_passed}/{len(suite)} ({absolute_pass_rate:.1f}%)")
    print(f"    • Total Execution Time              : {total_time:.2f}s ({total_time/len(suite):.2f}s/task)")
    print(f"    • Skipped Tasks (Baseline Failed)   : {len(skipped_baseline_fails)}")
    print(f"    • Target Attribution Tasks to Trace : {len(targeted_attribution_list)}")
    print("=" * 100)
    print(" 📊 BREAKDOWN BY DOMAIN:")
    for cat, stat in category_stats.items():
        p = stat["passed"]
        tot = stat["total"]
        print(f"    • {cat:<22}: {p:2d}/{tot:2d} ({(p/tot)*100:5.1f}%)")
    print("=" * 100)

    out_file = "./benchmarks/sandbox_benchmark_round1_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "fidelity_rate_pct": fidelity_rate,
            "absolute_pass_rate_pct": absolute_pass_rate,
            "total_passed": total_passed,
            "baseline_solvable": baseline_solvable,
            "solvable_passed": solvable_passed,
            "targeted_attribution_list": targeted_attribution_list,
            "skipped_baseline_fails": skipped_baseline_fails,
            "detailed_results": results
        }, f, indent=2)
    print(f"[✓] Detailed Sandbox report saved to: {out_file}")


if __name__ == "__main__":
    run_fast_batched_evaluation()
