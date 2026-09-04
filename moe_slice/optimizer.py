"""
Iterative Subnet Optimizer (Auto-Tune Optimization Loop).
Tự động tìm kiếm cấu hình Subnet tối ưu (Pareto-Optimal Frontier)
thông qua cơ chế Step-Wise Backward Elimination & Soft-Masking Validation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from moe_slice.config import SliceConfig
from moe_slice.hooks import MoERouterProfilerHook
from moe_slice.pruner import MoESubnetPruner
from moe_slice.exporter import MoEModelExporter
from moe_slice.profiler import CALIBRATION_DOMAINS


class IterativeSubnetOptimizer:
    """
    Vòng lặp tối ưu hóa tự động (Feedback Loop):
    - Tỉa bớt từng nhóm expert (Step-wise).
    - Sử dụng Soft-Masking để đo đạc tác động tức thì mà không cần tốn chi phí copy tensor.
    - Tự động hoàn tác (Rollback) nếu phát hiện Loss bị nhảy vọt (Spike).
    - Dừng lại tại điểm Pareto tối ưu (VRAM nhỏ nhất nhưng Accuracy cao nhất).
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: SliceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo tập active experts cho từng layer
        self.num_layers = 0
        self.initial_experts_per_layer: Dict[int, int] = {}
        self.active_experts_per_layer: Dict[int, Set[int]] = {}
        self._init_active_experts()

    def _init_active_experts(self):
        """Khởi tạo tập chuyên gia ban đầu (100% active)."""
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers

        self.num_layers = len(transformer_layers)
        for l_idx, layer in enumerate(transformer_layers):
            num_exp = 60
            if hasattr(layer, "mlp"):
                if hasattr(layer.mlp, "gate"):
                    num_exp = getattr(layer.mlp.gate, "num_experts", 60)
            elif hasattr(layer, "block_sparse_moe"):
                num_exp = getattr(layer.block_sparse_moe.gate, "num_experts", 60)
                
            self.initial_experts_per_layer[l_idx] = num_exp
            self.active_experts_per_layer[l_idx] = set(range(num_exp))

    def evaluate_validation_loss(self, validation_prompts: List[str]) -> float:
        """
        Đo đạc Loss trung bình trên tập validation prompts để đánh giá chất lượng mô hình.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for prompt in validation_prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                inputs["labels"] = inputs["input_ids"].clone()
                outputs = self.model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    count += 1

        return total_loss / max(1, count)

    def apply_soft_mask(self, masked_experts: Dict[int, Set[int]]) -> List[Any]:
        """
        Gắn PyTorch Forward Hook để tạm thời Soft-Mask (đặt logit = -1e9)
        cho các expert bị loại bỏ mà không làm thay đổi weights vật lý.
        """
        hook_handles = []
        
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers

        for l_idx, layer in enumerate(transformer_layers):
            if l_idx not in masked_experts or len(masked_experts[l_idx]) == 0:
                continue

            to_mask = list(masked_experts[l_idx])
            gate_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                gate_module = layer.mlp.gate
            elif hasattr(layer, "block_sparse_moe"):
                gate_module = layer.block_sparse_moe.gate

            if gate_module is not None:
                def make_mask_hook(mask_list):
                    def mask_fn(module, input_tensor, output_tensor):
                        # output_tensor có thể là logits hoặc tuple
                        if isinstance(output_tensor, tuple):
                            logits = output_tensor[0]
                            logits[..., mask_list] = -1e9
                            return (logits,) + output_tensor[1:]
                        else:
                            output_tensor[..., mask_list] = -1e9
                            return output_tensor
                    return mask_fn

                handle = gate_module.register_forward_hook(make_mask_hook(to_mask))
                hook_handles.append(handle)

        return hook_handles

    def optimize(
        self,
        calibration_prompts: List[str],
        validation_prompts: List[str],
        target_retention_ratio: float = 0.60,  # Mục tiêu giữ 60% experts (giảm 40% VRAM)
        step_prune_rate: float = 0.05,         # Mỗi vòng tỉa 5% experts
        max_loss_increase_pct: float = 8.0     # Ngưỡng tăng Loss tối đa cho phép (8%)
    ) -> Dict[int, List[int]]:
        """
        Thực thi toàn bộ vòng lặp tối ưu hóa Step-Wise.
        """
        print("=" * 75)
        print(" 🔄 BẮT ĐẦU VÒNG LẶP AUTO-TUNE (ITERATIVE SUBNET OPTIMIZATION)")
        print(f"    • Target Retention: {target_retention_ratio * 100:.1f}%")
        print(f"    • Step Pruning Rate: {step_prune_rate * 100:.1f}% per iteration")
        print(f"    • Max Allowed Loss Increase: +{max_loss_increase_pct:.1f}%")
        print("=" * 75)

        # 1. Đo Baseline Loss
        baseline_loss = self.evaluate_validation_loss(validation_prompts)
        print(f"[*] Baseline Validation Loss ban đầu: {baseline_loss:.4f}")

        current_retention = 1.0
        iteration = 1
        history = []

        total_experts_start = sum(len(exps) for exps in self.active_experts_per_layer.values())
        
        while current_retention > target_retention_ratio:
            print(f"\n--- [Iteration {iteration:02d}] Current Retention: {current_retention * 100:.1f}% ---")
            
            # Step A: Profile router activations trên tập active hiện tại
            hook = MoERouterProfilerHook(self.model)
            with torch.no_grad():
                for p in calibration_prompts[:10]:
                    inputs = self.tokenizer(p, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    _ = self.model(**inputs)
            
            # Step B: Xác định các candidate experts ít quan trọng nhất trên từng layer
            candidates_to_prune: Dict[int, Set[int]] = {}
            for l_idx in range(self.num_layers):
                A = hook.get_layer_activations(l_idx)
                importance = A.sum(dim=0).cpu().numpy()  # (Num_Experts,)
                
                # Chỉ xét các expert đang active
                active_list = sorted(list(self.active_experts_per_layer[l_idx]))
                active_importance = [(e, importance[e]) for e in active_list]
                active_importance.sort(key=lambda x: x[1])  # Sắp xếp tăng dần theo importance
                
                num_to_remove = max(1, int(np.round(self.initial_experts_per_layer[l_idx] * step_prune_rate)))
                # Không cắt dưới ngưỡng tối thiểu (ví dụ tối thiểu 16 experts)
                if len(active_list) - num_to_remove < 16:
                    candidates_to_prune[l_idx] = set()
                    continue
                    
                to_remove = set([x[0] for x in active_importance[:num_to_remove]])
                candidates_to_prune[l_idx] = to_remove

            hook.remove_hooks()

            # Step C: Virtual Soft-Masking để kiểm tra Loss
            # Tập bị mask = (toàn bộ các expert đã bị loại trước đó) + (candidates mới)
            all_masked: Dict[int, Set[int]] = {}
            for l_idx in range(self.num_layers):
                currently_inactive = set(range(self.initial_experts_per_layer[l_idx])) - self.active_experts_per_layer[l_idx]
                all_masked[l_idx] = currently_inactive | candidates_to_prune[l_idx]

            mask_handles = self.apply_soft_mask(all_masked)
            new_loss = self.evaluate_validation_loss(validation_prompts)
            
            # Dọn dẹp mask hooks
            for h in mask_handles:
                h.remove()

            loss_diff_pct = ((new_loss - baseline_loss) / baseline_loss) * 100.0
            print(f"  • Validation Loss sau khi tỉa thử: {new_loss:.4f} (Biến thiên: {loss_diff_pct:+.2f}%)")

            # Step D: Rollback hoặc Commit
            if loss_diff_pct > max_loss_increase_pct:
                print(f"  [⚠️ GUARD-RAIL TRIGGERED] Loss tăng vượt ngưỡng ({loss_diff_pct:+.2f}% > +{max_loss_increase_pct}%).")
                print(f"  [!] Đã tìm thấy ranh giới tối ưu (Optimal Pareto Knee-Point)! Dừng vòng lặp.")
                break
            else:
                # Commit: Chính thức loại bỏ candidates khỏi active set
                total_removed_this_step = 0
                for l_idx in range(self.num_layers):
                    self.active_experts_per_layer[l_idx] -= candidates_to_prune[l_idx]
                    total_removed_this_step += len(candidates_to_prune[l_idx])

                current_total_active = sum(len(exps) for exps in self.active_experts_per_layer.values())
                current_retention = current_total_active / total_experts_start
                print(f"  [✓ COMMIT] Tỉa thành công {total_removed_this_step} experts. Tổng số experts còn lại: {current_total_active}/{total_experts_start} ({current_retention * 100:.1f}%)")

                history.append([iteration, f"{current_retention * 100:.1f}%", f"{new_loss:.4f}", f"{loss_diff_pct:+.2f}%", "✅ PASS"])
                iteration += 1

        print("\n" + "=" * 75)
        print(" 🏆 BẢNG NHẬT KÝ TỐI ƯU HÓA AUTO-TUNE (PARETO OPTIMIZATION LOG)")
        print("=" * 75)
        print(tabulate(history, headers=["Vòng Lặp", "Tỷ Lệ Giữ Lại", "Val Loss", "Biến Thiên Loss", "Trạng Thái"], tablefmt="fancy_grid"))

        # Trả về danh sách selected experts tối ưu cho từng layer
        final_selected = {l_idx: sorted(list(exps)) for l_idx, exps in self.active_experts_per_layer.items()}
        return final_selected
