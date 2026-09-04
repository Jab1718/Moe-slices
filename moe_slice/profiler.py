"""
Layer-wise True Hidden States Profiler for Deep Sparse MoE.
Eliminates Feature Space Drift by capturing actual intermediate representations h_l
and projecting them onto original unpruned Router Gates W_gate[l].
"""

import json
import torch
from typing import Dict, List, Optional


class TrueLayerwiseProfiler:
    def __init__(self, num_layers: int = 48, total_experts: int = 512, target_experts: int = 160):
        self.num_layers = num_layers
        self.total_experts = total_experts
        self.target_experts = target_experts
        assert target_experts % 16 == 0, "Target experts count must be a multiple of 16 for Tensor Core alignment."

        self.expert_activation_counts = {
            l: torch.zeros(total_experts, dtype=torch.float32)
            for l in range(num_layers)
        }

    def record_layer_activations(self, layer_idx: int, true_hidden_state: torch.Tensor, gate_weight: torch.Tensor):
        """
        Projects actual hidden state h_l onto original Router Gate W_gate[l].
        true_hidden_state: [B, T, D]
        gate_weight: [total_experts, D]
        """
        with torch.no_grad():
            h_flat = true_hidden_state.view(-1, true_hidden_state.shape[-1]).to(gate_weight.device)
            # Logits: [N_tokens, total_experts]
            logits = h_flat @ gate_weight.T
            # Top-8 active experts per token
            _, top_k_indices = torch.topk(logits, k=8, dim=-1)
            
            # Accumulate activation counts
            counts = torch.bincount(top_k_indices.view(-1), minlength=self.total_experts).float()
            self.expert_activation_counts[layer_idx] += counts.cpu()

    def generate_expert_map(self) -> Dict[str, List[int]]:
        """
        Generates the target expert index map for all layers,
        strictly sorted and aligned to multiple of 16.
        """
        expert_map = {}
        for l in range(self.num_layers):
            counts = self.expert_activation_counts[l]
            _, top_indices = torch.topk(counts, k=self.target_experts)
            sorted_indices = sorted(top_indices.tolist())
            expert_map[str(l)] = sorted_indices
        return expert_map

    def save_expert_map(self, output_file: str):
        mapping = self.generate_expert_map()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"[✓] Saved Layer-wise True Hidden States expert map ({self.target_experts} exp/layer) to: {output_file}")
