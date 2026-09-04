"""
Hardware Inference & Latency Benchmarking (TTFT, Decode Throughput, Peak VRAM).
Tuân thủ tiêu chuẩn đo lường ms bằng torch.cuda.Event và warm-up GPU.
"""

import time
import argparse
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_inference(model_path: str, prompt: str = "Implement an asynchronous actor-based pipeline in Rust.", max_new_tokens: int = 256):
    """
    Đo lường chi tiết:
    - Time-to-First-Token (TTFT)
    - Decode Throughput (tokens/second)
    - Peak VRAM (GB)
    """
    print(f"\n" + "=" * 60)
    print(f" 🚀 BENCHMARKING MODEL: {model_path}")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu GPU NVIDIA để thực hiện benchmark phần cứng!")

    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.perf_counter() - load_start
    model.eval()

    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # 1. GPU WARM-UP (Ít nhất 2 lần trước khi ghi nhận số liệu)
    print("[*] Đang thực hiện GPU Warm-up (2 lần)...")
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    # 2. ĐO TIME-TO-FIRST-TOKEN (TTFT / Prefill Latency)
    print("[*] Đang đo Time-to-First-Token (TTFT)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        end_event.record()
    
    torch.cuda.synchronize()
    ttft_ms = start_event.elapsed_time(end_event)

    # 3. ĐO DECODE THROUGHPUT (Tokens per Second)
    print(f"[*] Đang đo Decode Throughput ({max_new_tokens} tokens)...")
    start_event.record()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    end_event.record()
    
    torch.cuda.synchronize()
    total_gen_ms = start_event.elapsed_time(end_event)
    
    total_tokens_generated = outputs.shape[1] - prompt_len
    decode_ms = total_gen_ms - ttft_ms
    decode_throughput = (total_tokens_generated - 1) / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

    # 4. GHI NHẬN PEAK VRAM
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    results = [
        ["Model Name / Path", model_path],
        ["Prompt Length (Tokens)", prompt_len],
        ["Generated Tokens", total_tokens_generated],
        ["Model Load Time", f"{load_time:.2f} s"],
        ["Time-to-First-Token (TTFT)", f"{ttft_ms:.2f} ms"],
        ["Decode Throughput", f"{decode_throughput:.2f} tokens/s"],
        ["Peak VRAM Allocation", f"{peak_vram_gb:.2f} GB"]
    ]

    print("\n" + tabulate(results, headers=["Metric", "Measured Value"], tablefmt="fancy_grid"))
    return {
        "ttft_ms": ttft_ms,
        "throughput_tps": decode_throughput,
        "peak_vram_gb": peak_vram_gb
    }


def main():
    parser = argparse.ArgumentParser(description="MoE Hardware Latency Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn model cần đo đạc")
    parser.add_argument("--prompt", type=str, default="Write a concurrent async web crawler in Rust with tokio.", help="Prompt kiểm thử")
    parser.add_argument("--tokens", type=int, default=256, help="Số token sinh ra")
    args = parser.parse_args()

    benchmark_inference(args.model, args.prompt, args.tokens)


if __name__ == "__main__":
    main()
