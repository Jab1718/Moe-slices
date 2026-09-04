"""
Command Line Interface (CLI) for moe-slice toolkit.
A Pure MoE Slicing & Subnet Extraction Framework.

Commands:
- moe-slice profile   : Capture layer-wise true hidden states and rank experts
- moe-slice plan      : Analyze co-activations and enforce hardware alignment (multiple of 16)
- moe-slice slice     : Execute zero-RAM streaming slicing from safetensors shards
- moe-slice attribute : Run closed-loop attribution tracing to diagnose neuron deficit
- moe-slice eval      : Run multi-lingual execution-based sandbox benchmark
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="moe-slice",
        description="moe-slice: High-Performance Slicing & Subnet Extraction Toolkit for Deep Sparse MoE LLMs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available MoE-Slice Lifecycle commands")

    # 1. Command: profile
    p_prof = subparsers.add_parser("profile", help="Profile expert activations using Layer-wise True Hidden States")
    p_prof.add_argument("--model-path", type=str, required=True, help="Path to base monolithic MoE model")
    p_prof.add_argument("--output-map", type=str, default="true_layerwise_160exp_map.json", help="Output path for expert map JSON")
    p_prof.add_argument("--target-experts", type=int, default=160, help="Number of experts to retain per layer (multiple of 16)")

    # 2. Command: plan
    p_plan = subparsers.add_parser("plan", help="Plan hardware-aligned matrix geometry (enforce multiple of 16)")
    p_plan.add_argument("--raw-map", type=str, required=True, help="Raw expert selection map")
    p_plan.add_argument("--align-multiple", type=int, default=16, help="GEMM alignment factor (default: 16)")
    p_plan.add_argument("--output-plan", type=str, default="aligned_slice_plan.json", help="Output aligned slice plan")

    # 3. Command: slice
    p_slice = subparsers.add_parser("slice", help="Stream and slice checkpoint shards with Zero-RAM overhead")
    p_slice.add_argument("--source-dir", type=str, required=True, help="Directory containing source safetensors shards")
    p_slice.add_argument("--output-dir", type=str, required=True, help="Directory to save sliced checkpoint")
    p_slice.add_argument("--expert-map", type=str, required=True, help="Path to aligned expert map JSON")
    p_slice.add_argument("--align-multiple", type=int, default=16, help="Hardware alignment multiple")

    # 4. Command: attribute
    p_attr = subparsers.add_parser("attribute", help="Run Closed-Loop Attribution Tracing on evaluation failures")
    p_attr.add_argument("--expert-map", type=str, required=True, help="Retained expert map JSON")
    p_attr.add_argument("--failures-file", type=str, required=True, help="Path to evaluation failure trace file")

    # 5. Command: eval
    p_eval = subparsers.add_parser("eval", help="Run multi-lingual execution-based sandbox benchmark")
    p_eval.add_argument("--model-path", type=str, required=True, help="Path to sliced checkpoint to evaluate")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"[*] moe-slice executing lifecycle stage: {args.command.upper()}")


if __name__ == "__main__":
    main()
