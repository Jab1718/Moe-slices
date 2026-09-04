"""
Visualization script for MoE Co-Activation Matrices and Layer-by-Layer Activation Heatmaps.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_slice.config import SliceConfig
from moe_slice.profiler import CoActivationProfiler, CALIBRATION_DOMAINS


def plot_coactivation_heatmaps(profile_data: dict, output_dir: str = "./benchmark_outputs"):
    """
    Vẽ biểu đồ Heatmap cho ma trận tương quan giữa các experts trên các tầng tiêu biểu.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers = profile_data["num_layers"]
    domain = profile_data["domain"]
    
    # Chọn 3 layer đại diện: Early, Middle, Late
    sample_layers = [
        int(num_layers * 0.15),  # Early
        int(num_layers * 0.50),  # Middle
        int(num_layers * 0.85)   # Late
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"MoE Expert Co-Activation Heatmaps (Domain: {domain.upper()})", fontsize=16, fontweight="bold")
    
    for idx, l_idx in enumerate(sample_layers):
        data = profile_data["layers"][str(l_idx)] if str(l_idx) in profile_data["layers"] else profile_data["layers"][l_idx]
        matrix = np.array(data["coactivation_matrix"])
        
        ax = axes[idx]
        sns.heatmap(matrix, cmap="viridis", ax=ax, cbar=(idx == 2), square=True)
        layer_type = "Early (Syntax)" if idx == 0 else ("Middle (Specialization)" if idx == 1 else "Late (Format)")
        ax.set_title(f"Layer {l_idx} - {layer_type}", fontsize=12)
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Expert Index")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"coactivation_heatmap_{domain}.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ Heatmap Co-Activation tại: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MoE Co-Activation Patterns")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B-Chat", help="Tên model")
    parser.add_argument("--domain", type=str, default="coding", choices=["coding", "math", "general"])
    parser.add_argument("--output-dir", type=str, default="./benchmark_outputs")
    args = parser.parse_args()

    config = SliceConfig(
        model_name_or_path=args.model,
        domain=args.domain,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[*] Đang tải mô hình {config.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    profiler = CoActivationProfiler(model, tokenizer, config)
    profile_data = profiler.run()
    profiler.cleanup()

    plot_coactivation_heatmaps(profile_data, args.output_dir)


if __name__ == "__main__":
    main()
