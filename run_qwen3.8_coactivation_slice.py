"""
Streaming Sharded Slicer using 2D Co-Activation & Saliency Selection Map.
Cuts Qwen/Qwen3.8-Flash-Next (335GB) down to a pristine 73GB BF16 Subnet (128 experts/layer)
by extracting the optimal algorithmic co-activation bundles per layer.
All code, comments, and logging strictly in 100% English.
"""

import os
import gc
import json
import time
import shutil
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def get_layer_index_from_param(param_name: str) -> int:
    parts = param_name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return -1


def run_coactivation_streaming_slice(
    repo_id: str = "Qwen/Qwen3.8-Flash-Next",
    output_dir: str = "./qwen3.8_flash_bf16_73gb",
    selection_map_path: str = "./true_layerwise_160exp_map.json",
    clean_cache: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print(" 🍰 ELASTIC SPECIFICITY-PINNED STREAMING MOE SLICER: QWEN3.8-FLASH-NEXT (335GB -> 75GB)")
    print(f"    • Source Model Repo : {repo_id}")
    print(f"    • Selection Map     : {selection_map_path}")
    print(f"    • Output Directory  : {output_dir}")
    print(f"    • Architecture Mode : Pure Qwen4ExpForCausalLM (Stripping 95GB N-Gram PLE)")
    print("=" * 85)

    # 1. Load Selection Map
    with open(selection_map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    selection_map = {int(k): torch.tensor(v, dtype=torch.long) for k, v in raw_map.items()}
    num_exp_target = len(selection_map[0])
    print(f"[✓] Successfully loaded Elastic Selection Map: {num_exp_target} experts/layer across {len(selection_map)} layers.")

    # 2. Download Tokenizer, Generation Config, Config files
    print("\n[Step 1/4] Downloading Metadata, Configs, and Tokenizer...")
    config_files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"
    ]
    for fname in config_files:
        try:
            loc = hf_hub_download(repo_id=repo_id, filename=fname)
            dest = os.path.join(output_dir, fname)
            shutil.copyfile(loc, dest)
        except Exception:
            pass

    # Update config.json to elastic capacity (144 experts) and pure MoE architecture
    config_dest = os.path.join(output_dir, "config.json")
    if os.path.exists(config_dest):
        with open(config_dest, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["num_experts"] = num_exp_target
        cfg["architectures"] = ["Qwen4ExpForCausalLM"]
        cfg["ple_layer_ids"] = []
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            cfg["text_config"]["num_experts"] = num_exp_target
            cfg["text_config"]["architectures"] = ["Qwen4ExpForCausalLM"]
            cfg["text_config"]["ple_layer_ids"] = []
        with open(config_dest, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[✓] Updated config.json: num_experts = {num_exp_target}, ple_layer_ids = [], architecture = Qwen4ExpForCausalLM.")

    # 3. Load Index JSON
    print("\n[Step 2/4] Analyzing Safetensors Index Map from upstream...")
    index_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    all_shards = sorted(list(set(weight_map.values())))
    total_shards = len(all_shards)
    print(f"[✓] Total Upstream Shards: {total_shards} shards.")

    # 4. Stream and Slice Shards
    print("\n[Step 3/4] Commencing Elastic Streaming Slicing (Stripping 95GB N-Gram PLE)...")
    temp_weight_map = {}
    saved_shards = []
    total_orig_bytes = 0
    total_new_bytes = 0
    t_start = time.perf_counter()

    for s_idx, shard_name in enumerate(all_shards, 1):
        t0 = time.perf_counter()
        cached_shard = hf_hub_download(repo_id=repo_id, filename=shard_name)
        state_dict = load_file(cached_shard)

        sliced_state_dict = {}
        for param_name, tensor in state_dict.items():
            orig_bytes = tensor.numel() * tensor.element_size()
            total_orig_bytes += orig_bytes

            # 1. Strip N-Gram PLE embedding keys immediately
            if "ngram_embedding" in param_name:
                continue

            # 2. Slice MoE Experts according to Elastic Specificity Map
            is_expert_tensor = (
                ("experts.gate_up_proj" in param_name or "experts.down_proj" in param_name or "gate.weight" in param_name)
                and tensor.shape[0] == 512
            )

            layer_idx = get_layer_index_from_param(param_name)

            if is_expert_tensor and layer_idx in selection_map:
                idx_tensor = selection_map[layer_idx]
                sliced_t = tensor[idx_tensor].clone()
            elif is_expert_tensor:
                sliced_t = tensor[:num_exp_target].clone()
            else:
                sliced_t = tensor.clone()

            clean_name = param_name.replace("model.language_model.", "model.")
            sliced_state_dict[clean_name] = sliced_t
            t_bytes = sliced_t.numel() * sliced_t.element_size()
            total_new_bytes += t_bytes
            temp_weight_map[clean_name] = shard_name

        del state_dict
        gc.collect()

        # Save sliced shard only if it contains core tensors (ignore pure N-gram shards)
        if len(sliced_state_dict) > 0:
            out_shard_file = os.path.join(output_dir, shard_name)
            save_file(sliced_state_dict, out_shard_file)
            saved_shards.append(shard_name)

        del sliced_state_dict
        gc.collect()

        if clean_cache and os.path.exists(cached_shard):
            try:
                os.remove(cached_shard)
            except Exception:
                pass

        if s_idx % 10 == 0 or s_idx == total_shards:
            elapsed = time.perf_counter() - t0
            print(f"  [{s_idx:03d}/{total_shards:03d}] Shard: {shard_name} | Elapsed: {elapsed:.2f}s | Progress: {(s_idx/total_shards)*100:.1f}%", flush=True)

    # 5. Consolidate into 2 Fast Memory-Mapped Shards
    print("\n[Step 4/4] Consolidating sliced shards into 2 fast memory-mapped safetensors...")
    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    shard1_tensors = {}
    shard2_tensors = {}
    final_weight_map = {}

    half = len(saved_shards) // 2
    print("  • Consolidating Shard 1/2...")
    for s_name in saved_shards[:half]:
        sp = os.path.join(output_dir, s_name)
        if os.path.exists(sp):
            sd = load_file(sp)
            for k, v in sd.items():
                shard1_tensors[k] = v
                final_weight_map[k] = shard1_name
            del sd
            os.remove(sp)

    out1 = os.path.join(output_dir, shard1_name)
    save_file(shard1_tensors, out1)
    del shard1_tensors
    gc.collect()

    print("  • Consolidating Shard 2/2...")
    for s_name in saved_shards[half:]:
        sp = os.path.join(output_dir, s_name)
        if os.path.exists(sp):
            sd = load_file(sp)
            for k, v in sd.items():
                shard2_tensors[k] = v
                final_weight_map[k] = shard2_name
            del sd
            os.remove(sp)

    out2 = os.path.join(output_dir, shard2_name)
    save_file(shard2_tensors, out2)
    del shard2_tensors
    gc.collect()

    # Save final index file
    new_index_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(new_index_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"total_size": total_new_bytes},
            "weight_map": final_weight_map
        }, f, indent=2)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 85)
    print(" 🎉 CO-ACTIVATION SLICING COMPLETED SUCCESSFULLY!")
    print(f"    • Original 512-Expert Model Size : {total_orig_bytes / (1024**3):.2f} GB")
    print(f"    • New Pristine Co-Activated Model: {total_new_bytes / (1024**3):.2f} GB (~73 GB BF16)")
    print(f"    • Total Slicing Time             : {total_time / 60:.2f} minutes")
    print(f"    • Output Location                : {output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    run_coactivation_streaming_slice()
