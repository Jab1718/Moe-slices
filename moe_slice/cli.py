"""
Command Line Interface (CLI) for moe-slice toolkit.
Commands:
- moe-slice profile : Run layer-wise true hidden states profiling
- moe-slice slice   : Perform hardware-aligned streaming slicing
- moe-slice distill : Run Router-Gate DoRA distillation
- moe-slice eval    : Run multi-lingual execution-based sandbox benchmark
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="moe-slice",
        description="High-Performance Toolkit for Slicing, Profiling, and Calibrating Deep Sparse MoE LLMs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: profile
    p_prof = subparsers.add_parser("profile", help="Run layer-wise true hidden states profiler")
    p_prof.add_argument("--model-path", type=str, required=True, help="Path to base MoE model")
    p_prof.add_argument("--output-map", type=str, default="true_layerwise_160exp_map.json", help="Path to output expert map")
    p_prof.add_argument("--target-experts", type=int, default=160, help="Number of experts to retain per layer (multiple of 16)")

    # Command: slice
    p_slice = subparsers.add_parser("slice", help="Slice MoE checkpoint with expert map")
    p_slice.add_argument("--source-dir", type=str, required=True, help="Directory containing source safetensors")
    p_slice.add_argument("--output-dir", type=str, required=True, help="Directory to save sliced checkpoint")
    p_slice.add_argument("--expert-map", type=str, required=True, help="Path to expert map JSON")

    # Command: distill
    p_dist = subparsers.add_parser("distill", help="Run DoRA Router-Gate distillation")
    p_dist.add_argument("--model-path", type=str, required=True, help="Path to sliced checkpoint")
    p_dist.add_argument("--dataset", type=str, required=True, help="Path to distillation corpus JSONL")
    p_dist.add_argument("--epochs", type=int, default=1, help="Number of distillation epochs")
    p_dist.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")

    # Command: eval
    p_eval = subparsers.add_parser("eval", help="Run 100-task sandbox evaluation")
    p_eval.add_argument("--model-path", type=str, required=True, help="Path to model to evaluate")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"[*] moe-slice executing command: {args.command}")


if __name__ == "__main__":
    main()
