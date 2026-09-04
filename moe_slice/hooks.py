"""
Non-intrusive PyTorch Hooks for capturing MoE router logits and activations.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class MoERouterProfilerHook:
    """
    Hook gắn vào các tầng Gate / Router của mô hình MoE để thu thập:
    1. Router logits & probabilities cho từng token và từng expert.
    2. Tần suất chọn expert (Top-K selection counts).
    3. Trọng số đóng góp tích luỹ của từng expert.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[Any] = []
        # Dữ liệu lưu trữ: layer_idx -> List[Tensor] (batch_tokens, num_experts)
        self.activation_records: Dict[int, List[torch.Tensor]] = {}
        # Thống kê tổng hợp: layer_idx -> (num_experts,)
        self.expert_usage_counts: Dict[int, torch.Tensor] = {}
        self.expert_prob_sums: Dict[int, torch.Tensor] = {}
        
        self.num_layers = 0
        self.num_experts_per_layer: Dict[int, int] = {}
        self._register_hooks()

    def _find_moe_layers(self) -> List[Tuple[int, nn.Module, str]]:
        """
        Tự động phát hiện các tầng MoE Router/Gate trên nhiều kiến trúc khác nhau
        (Qwen2.5-MoE, Qwen1.5-MoE, DeepSeek-MoE, Mixtral).
        """
        moe_layers = []
        
        # Duyệt qua các layers của model
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers
            
        if transformer_layers is None:
            raise ValueError("Không tìm thấy danh sách layers trong kiến trúc mô hình!")

        for layer_idx, layer in enumerate(transformer_layers):
            # Qwen-MoE / Mixtral structure
            gate_module = None
            gate_type = ""
            
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "gate"):
                    gate_module = mlp.gate
                    gate_type = "standard_gate"
                elif hasattr(mlp, "router"):
                    gate_module = mlp.router
                    gate_type = "router"
            elif hasattr(layer, "block_sparse_moe"):
                gate_module = layer.block_sparse_moe.gate
                gate_type = "mixtral_gate"
                
            if gate_module is not None:
                moe_layers.append((layer_idx, gate_module, gate_type))
                
        return moe_layers

    def _register_hooks(self):
        """
        Đăng ký forward hook vào từng tầng gate.
        """
        moe_layers = self._find_moe_layers()
        self.num_layers = len(moe_layers)
        
        for layer_idx, gate_module, _ in moe_layers:
            self.activation_records[layer_idx] = []
            
            def make_hook(l_idx: int):
                def hook_fn(module, input_tensor, output_tensor):
                    with torch.no_grad():
                        # Output của gate thường là router logits (batch_size, seq_len, num_experts)
                        # hoặc tuple (router_logits, selected_experts)
                        logits = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                        
                        # Reshape về (-1, num_experts)
                        flat_logits = logits.view(-1, logits.shape[-1])
                        num_exp = flat_logits.shape[-1]
                        self.num_experts_per_layer[l_idx] = num_exp
                        
                        # Tính Softmax probabilities
                        probs = torch.softmax(flat_logits.float(), dim=-1).cpu()
                        self.activation_records[l_idx].append(probs)
                        
                        # Cập nhật thống kê tích lũy
                        if l_idx not in self.expert_prob_sums:
                            self.expert_prob_sums[l_idx] = probs.sum(dim=0)
                        else:
                            self.expert_prob_sums[l_idx] += probs.sum(dim=0)
                            
                return hook_fn

            handle = gate_module.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)

    def get_layer_activations(self, layer_idx: int) -> torch.Tensor:
        """
        Gộp toàn bộ tensor activation ghi nhận được ở tầng layer_idx thành 1 ma trận (Total_Tokens, Num_Experts).
        """
        if layer_idx not in self.activation_records or len(self.activation_records[layer_idx]) == 0:
            raise ValueError(f"Không có dữ liệu activation cho layer {layer_idx}")
        return torch.cat(self.activation_records[layer_idx], dim=0)

    def clear(self):
        """
        Dọn dẹp bộ nhớ đệm đã lưu.
        """
        for l_idx in self.activation_records:
            self.activation_records[l_idx].clear()
        self.expert_prob_sums.clear()
        self.expert_usage_counts.clear()

    def remove_hooks(self):
        """
        Gỡ bỏ toàn bộ hook ra khỏi model để tránh rò rỉ bộ nhớ.
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
