"""
Weight Extraction & MoE Subnet Pruning Engine.
Hỗ trợ cả Qwen2MoeTopKRouter, Batched 3D Tensor Experts, và ModuleList Experts.
"""

from typing import Dict, List, Tuple, Any
import copy
import torch
import torch.nn as nn
from moe_slice.config import SliceConfig


class MoESubnetPruner:
    """
    Module thực hiện cắt tỉa thực tế trên trọng số PyTorch của mô hình MoE:
    - Giữ nguyên 100% Shared Experts và Attention Backbone.
    - Trích xuất chỉ các FFN expert tensors được chọn trong Subnet.
    - Resize Router/Gate projection weights tương ứng với số lượng expert mới.
    """

    def __init__(self, model: nn.Module, config: SliceConfig):
        self.model = model
        self.config = config

    def _get_moe_layers(self) -> List[Tuple[int, nn.Module]]:
        """
        Tìm và trả về tất cả các tầng MoE (hỗ trợ Qwen2Moe, Qwen4Exp/Qwen3.8-Flash-Next, Mixtral, DeepSeek).
        """
        moe_layers = []
        transformer_layers = None
        
        # 1. Hỗ trợ Qwen4Exp (Qwen3.8-Flash-Next) Vision-Language & Text Model
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "text_model") and hasattr(self.model.text_model, "layers"):
            transformer_layers = self.model.text_model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers

        if transformer_layers is None:
            raise ValueError("Không tìm thấy transformer layers trong cấu trúc mô hình!")

        for l_idx, layer in enumerate(transformer_layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                moe_layers.append((l_idx, layer.mlp))
            elif hasattr(layer, "block_sparse_moe"):
                moe_layers.append((l_idx, layer.block_sparse_moe))
                
        return moe_layers

    def prune(self, selected_experts_per_layer: Dict[int, List[int]]) -> nn.Module:
        """
        Thực hiện cắt tỉa in-place trên mô hình và trả về mô hình đã rút gọn.
        """
        print("[*] Bắt đầu cắt tỉa trọng số mô hình theo cấu hình Subnet...")
        
        # Xác định danh sách layers
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers
            
        if transformer_layers is None:
            raise ValueError("Không tìm thấy layers trong cấu trúc mô hình!")

        for layer_idx, layer in enumerate(transformer_layers):
            if layer_idx not in selected_experts_per_layer:
                continue

            selected = selected_experts_per_layer[layer_idx]
            num_new_experts = len(selected)

            # 1. Xử lý kiến trúc Qwen-MoE / DeepSeek-MoE
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                mlp = layer.mlp
                experts_obj = mlp.experts
                
                # Case A: Qwen2MoeExperts (3D Batched Tensor: [num_experts, out_dim, in_dim])
                if hasattr(experts_obj, "gate_up_proj") and hasattr(experts_obj, "down_proj"):
                    device = experts_obj.gate_up_proj.device
                    idx_tensor = torch.tensor(selected, dtype=torch.long, device=device)
                    
                    new_gate_up = experts_obj.gate_up_proj.data[idx_tensor]
                    new_down = experts_obj.down_proj.data[idx_tensor]
                    
                    experts_obj.gate_up_proj = nn.Parameter(new_gate_up)
                    experts_obj.down_proj = nn.Parameter(new_down)
                    if hasattr(experts_obj, "num_experts"):
                        experts_obj.num_experts = num_new_experts

                # Case B: Standard nn.ModuleList
                elif isinstance(experts_obj, nn.ModuleList):
                    new_experts = nn.ModuleList([experts_obj[i] for i in selected])
                    mlp.experts = new_experts
                    
                # Resize Gate Router Matrix
                if hasattr(mlp, "gate"):
                    old_gate = mlp.gate
                    device = old_gate.weight.device
                    idx_tensor = torch.tensor(selected, dtype=torch.long, device=device)
                    new_gate_weight = old_gate.weight.data[idx_tensor, :]
                    
                    old_gate.weight = nn.Parameter(new_gate_weight)
                    
                    if hasattr(old_gate, "bias") and old_gate.bias is not None:
                        old_gate.bias = nn.Parameter(old_gate.bias.data[idx_tensor])
                        
                    if hasattr(old_gate, "num_experts"):
                        old_gate.num_experts = num_new_experts
                    if hasattr(old_gate, "out_features"):
                        old_gate.out_features = num_new_experts
                    if hasattr(old_gate, "top_k"):
                        old_gate.top_k = min(old_gate.top_k, num_new_experts)
                        
                if hasattr(mlp, "num_experts"):
                    mlp.num_experts = num_new_experts

            # 2. Xử lý kiến trúc Mixtral (block_sparse_moe)
            elif hasattr(layer, "block_sparse_moe"):
                moe = layer.block_sparse_moe
                old_experts = moe.experts
                new_experts = nn.ModuleList([old_experts[i] for i in selected])
                moe.experts = new_experts
                
                old_gate = moe.gate
                idx_tensor = torch.tensor(selected, dtype=torch.long, device=old_gate.weight.device)
                new_gate_weight = old_gate.weight.data[idx_tensor, :]
                old_gate.weight = nn.Parameter(new_gate_weight)
                if hasattr(old_gate, "bias") and old_gate.bias is not None:
                    old_gate.bias = nn.Parameter(old_gate.bias.data[idx_tensor])
                if hasattr(old_gate, "num_experts"):
                    old_gate.num_experts = num_new_experts
                if hasattr(old_gate, "out_features"):
                    old_gate.out_features = num_new_experts
                moe.num_experts = num_new_experts

        print(f"[✓] Hoàn tất cắt tỉa. Mô hình hiện đã được thu gọn thành Subnet chuyên biệt.")
        return self.model
