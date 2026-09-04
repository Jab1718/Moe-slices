"""
Quick 10-Task Sandbox Benchmark for Qwen3.8-Flash-Coder (160 Experts M2 Subnet).
Runs synchronously in foreground with streaming real-time output.
All code, comments, and logging strictly in 100% English.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import sys
import time
import json
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.eval_comprehensive_100_sandbox import build_100_sandbox_tasks


def clean_python_code(prompt: str, generated_text: str) -> str:
    """Extracts clean python code from model generation."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", generated_text, flags=re.IGNORECASE).strip()
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

    if prompt.strip() not in content and any(kw in content for kw in ["def ", "class ", "import "]):
        return content
    return f"{prompt.strip()}\n    {content}" if not content.startswith(prompt.strip()) else content


def execute_in_sandbox(full_code: str, test_assertions: str, timeout: float = 3.0) -> bool:
    """Executes code in isolated python subprocess."""
    script = f"{full_code}\n\n# --- ASSERTIONS ---\n{test_assertions}\n"
    try:
        compile(script, "<sandbox>", "exec")
    except Exception:
        return False

    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return res.returncode == 0
    except Exception:
        return False


def run_quick_10():
    model_path = "./qwen3.8_flash_bf16_73gb"
    print("=" * 90, flush=True)
    print(" ⚡ QUICK 10-TASK SANDBOX BENCHMARK (M2 SUBSET: 160 EXPERTS/LAYER)", flush=True)
    print(f"    • Model Tested    : {model_path}", flush=True)
    print(f"    • Hardware Target : 3x RTX 5000 Ada (CUDA: 0, 2, 3)", flush=True)
    print(f"    • Execution Mode  : Synchronous Foreground with Real-Time Feedback", flush=True)
    print("=" * 90, flush=True)

    # 1. Load Tokenizer
    print("\n[*] Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Partition 48 Layers across 3 GPUs (Dynamic 14-17-17 for zero OOM)
    # GPU 0 (26.8GB avail): 14 layers + Embed (~24.2GB) -> Headroom ~2.6GB!
    # GPU 2 / cuda:1 (29.7GB avail): 17 layers (~28.5GB) -> Headroom ~1.2GB!
    # GPU 3 / cuda:2 (31.6GB avail): 17 layers + Norm + LM_Head (~29.2GB) -> Headroom ~2.4GB!
    device_map = {
        "model.embed_tokens": "cuda:0",
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

    print("[*] Loading 81.9GB weights onto 3 GPUs (BF16 Precision, 14-17-17)...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    load_time = time.perf_counter() - t0
    print(f"[✓] Model weights loaded into 3 GPUs in {load_time:.2f}s!", flush=True)

    # 3. Warm-up GPU
    print("[*] Performing GPU Warm-up pass...", flush=True)
    warm_in = tokenizer("def warm_up(): return True", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**warm_in, max_new_tokens=10)
    print("[✓] GPU Warm-up completed!\n", flush=True)

    # 4. Select 10 Representative Tasks across all core domains
    full_suite = build_100_sandbox_tasks()
    selected_indices = [0, 5, 10, 15, 25, 35, 45, 55, 65, 85]
    tasks = [full_suite[i] for i in selected_indices if i < len(full_suite)]

    print(f"[*] Commencing Real-Time Evaluation on {len(tasks)} Key Representative Tasks:", flush=True)
    print("-" * 90, flush=True)

    passed_count = 0
    results = []

    for idx, task in enumerate(tasks, 1):
        tid = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        asserts = "\n".join(task.get("asserts", []))

        # Format with direct code execution (Bypassing long thinking loop)
        messages = [
            {"role": "system", "content": "You are an automated Python code generator. Complete the code function directly. Do NOT explain."},
            {"role": "user", "content": f"Complete this Python function:\n```python\n{prompt}\n```"}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if formatted.endswith("<think>\n"):
            formatted += "</think>\n```python\n"
        elif not "<think>" in formatted:
            formatted += "<think>\n</think>\n```python\n"
        else:
            formatted += "```python\n"

        inputs = tokenizer(formatted, return_tensors="pt").to("cuda:0")

        t_gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen_time = time.perf_counter() - t_gen_start
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Prepend ```python\n if model continued from the injected fence
        clean_code = clean_python_code(prompt, "```python\n" + gen_text)

        # Sandbox verification
        is_pass = execute_in_sandbox(clean_code, asserts, timeout=3.0)
        if is_pass:
            passed_count += 1
            status_str = "✅ PASS"
        else:
            status_str = "❌ FAIL"

        print(f"  [{idx:02d}/10] {status_str} | Domain: {cat:<22} | Task: {tid:<28} | Time: {gen_time:.2f}s", flush=True)
        if not is_pass:
            lines = clean_code.strip().split("\n")
            preview = lines[1] if len(lines) > 1 else lines[0]
            print(f"        👉 Code Snippet: {preview.strip()}", flush=True)

        results.append({
            "id": tid,
            "category": cat,
            "pass": is_pass,
            "gen_time": gen_time,
            "generated_code": clean_code
        })

    # Summary
    print("-" * 90)
    pass_rate = (passed_count / len(tasks)) * 100.0
    print("=" * 90)
    print(f" 🏆 QUICK EVALUATION FINISHED: {passed_count}/{len(tasks)} PASSED ({pass_rate:.1f}%)")
    print(f"    • Model Subnet Tested : 160 Experts / Layer (Pure Qwen4ExpForCausalLM)")
    print(f"    • Total Eval Time     : {sum(r['gen_time'] for r in results):.2f}s")
    print("=" * 90)


if __name__ == "__main__":
    run_quick_10()
