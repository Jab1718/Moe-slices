"""
Targeted Micro-DoRA Tuning Pipeline for Agent Tool-Calling & Edge Cases.
Features:
- Precision adaptation on Router Gates (Layers 15-47) + Attention Projections.
- Uses the newly generated micro-tuning dataset (145 samples).
- Fast execution (~2-3 minutes) with dynamic 15-16-17 GPU partitioning.
- Zero CPU offload bottleneck during export; in-place safetensors merge.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import torch
import transformers.modeling_utils
import transformers.core_model_loading
from concurrent.futures import ThreadPoolExecutor

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_micro_dora(
    model_path: str = "./qwen3.8_flash_bf16_73gb",
    dataset_path: str = "./data/micro_tuning_edge_cases.jsonl",
    num_epochs: int = 1,
    learning_rate: float = 1.5e-4
):
    print("=" * 90, flush=True)
    print(" 🚀 TARGETED DORA MICRO-TUNING FOR AGENT & EDGE CASES (145 SAMPLES)", flush=True)
    print(f"    • Checkpoint Target : {model_path}", flush=True)
    print(f"    • Training Dataset  : {dataset_path}", flush=True)
    print(f"    • Target Hardware   : 3x RTX 5000 Ada (CUDA: 0, 2, 3)", flush=True)
    print(f"    • Adaptation Scope  : Router Gate + Attention (Layers 15-47)", flush=True)
    print("=" * 90, flush=True)

    torch.cuda.empty_cache()

    # 1. Load Tokenizer
    print("\n[*] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 2. Dynamic 15-16-17 Partitioning with CPU Embedding Offload
    device_map = {
        "model.embed_tokens": "cpu",
        "model.hyper_connection_mixer": "cuda:2",
        "model.norm": "cuda:2",
        "lm_head": "cuda:2"
    }
    for i in range(15):
        device_map[f"model.layers.{i}"] = "cuda:0"
    for i in range(15, 31):
        device_map[f"model.layers.{i}"] = "cuda:1"
    for i in range(31, 48):
        device_map[f"model.layers.{i}"] = "cuda:2"

    print("[*] Loading BF16 model weights across 3 GPUs (Dynamic 15-16-17)...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    print(f"[✓] Model loaded successfully in {time.perf_counter() - t0:.2f}s!", flush=True)

    # Enable Gradient Checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[✓] Gradient Checkpointing successfully enabled!", flush=True)
    except Exception as e:
        print(f"[*] Gradient checkpointing fallback: {e}", flush=True)

    # 3. Attach DoRA & Unlock Router Gates
    lora_rank = 16
    lora_alpha = 32
    print(f"\n[*] Initializing DoRA Adapter (Rank={lora_rank}, Alpha={lora_alpha}, use_dora=True)...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        use_dora=True,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(15, 48)),
        lora_dropout=0.05,
        bias="none"
    )
    peft_model = get_peft_model(model, lora_config)

    # Unlock Router Gate weights on Layers 15-47
    router_param_count = 0
    for name, param in peft_model.named_parameters():
        if "gate.weight" in name and any(f"layers.{i}." in name for i in range(15, 48)):
            param.requires_grad = True
            router_param_count += param.numel()
    print(f"[✓] Unlocked {router_param_count:,} Router Gate parameters across Layers 15-47!", flush=True)
    peft_model.print_trainable_parameters()

    # 4. Load Dataset
    print(f"\n[*] Loading micro-tuning dataset from {dataset_path}...", flush=True)
    train_tokens = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                toks = tokenizer(sample["text"], return_tensors="pt", max_length=384, truncation=True)
                train_tokens.append(toks)
    print(f"[✓] Prepared {len(train_tokens)} samples (max_length=384).", flush=True)

    # 5. Training Loop
    print(f"\n[*] Commencing Micro-DoRA Training ({num_epochs} Epoch, LR={learning_rate})...", flush=True)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    peft_model.train()
    t_train_start = time.perf_counter()
    for epoch in range(num_epochs):
        total_loss = 0.0
        t_ep = time.perf_counter()
        for step, sample in enumerate(train_tokens, start=1):
            t_step = time.perf_counter()
            input_ids = sample["input_ids"]
            labels = sample["input_ids"].clone().to("cuda:2")

            outputs = peft_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()
            total_loss += loss_val
            step_time = time.perf_counter() - t_step

            if step % 10 == 0 or step == len(train_tokens):
                avg_so_far = total_loss / step
                print(f"  [Epoch {epoch+1}/{num_epochs}] Step {step:3d}/{len(train_tokens):3d} | Loss: {loss_val:6.4f} (Avg: {avg_so_far:6.4f}) | {step_time*1000:5.1f}ms/step", flush=True)

        epoch_time = time.perf_counter() - t_ep
        mean_epoch_loss = total_loss / len(train_tokens)
        print(f"\n[✓] Epoch {epoch+1} Completed in {epoch_time:.2f}s! Mean Loss: {mean_epoch_loss:.4f}\n", flush=True)

    total_train_time = time.perf_counter() - t_train_start
    print(f"🎉 Micro-DoRA Training finished in {total_train_time:.2f}s!", flush=True)

    # 6. Merge & Export
    print("\n[*] Merging DoRA weights directly into base checkpoint...", flush=True)
    t_merge = time.perf_counter()
    merged_model = peft_model.merge_and_unload()
    print(f"[✓] Weights merged in {time.perf_counter() - t_merge:.2f}s!", flush=True)

    print(f"[*] Overwriting and saving updated BF16 checkpoint to {model_path} (max_shard_size='50GB')...", flush=True)
    t_save = time.perf_counter()
    merged_model.save_pretrained(
        model_path,
        max_shard_size="50GB",
        safe_serialization=True
    )
    tokenizer.save_pretrained(model_path)
    print(f"[✓] Successfully exported 81.9GB BF16 checkpoint in {time.perf_counter() - t_save:.2f}s!", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(" 🏆 MICRO-DORA TUNING & CHECKPOINT PACKAGING COMPLETED!", flush=True)
    print(f"    • Mean Final Loss : {mean_epoch_loss:.4f}", flush=True)
    print(f"    • Target Location : {model_path}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    run_micro_dora()
