"""
Graph Clustering & Layer-Adaptive Knapsack Selection for MoE Experts.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
from moe_slice.config import SliceConfig


class CoActivationClusterer:
    """
    Thuật toán phân cụm đồ thị và chọn lọc bó chuyên gia (Co-Activation Bundles):
    - Đảm bảo các chuyên gia có tính tương quan cao (ví dụ Cú pháp + Logic) không bị tách rời.
    - Phân bổ ngân sách linh hoạt theo độ sâu tầng Transformer (Early / Middle / Late).
    """

    def __init__(self, config: SliceConfig):
        self.config = config

    def compute_layer_expert_budgets(self, num_layers: int, num_experts_per_layer: Dict[int, int]) -> Dict[int, int]:
        """
        Tính số lượng chuyên gia đồng nhất (Uniform Budget) cho mọi tầng
        để đảm bảo tương thích 100% với HuggingFace / vLLM chuẩn.
        """
        rule = self.config.layer_budget
        target_ratio = rule.middle_layers_ratio  # 0.40 (40% experts = 24/60 experts)
        
        layer_budgets = {}
        for l_idx in range(num_layers):
            total_exp = num_experts_per_layer.get(l_idx, 60)
            target_k = max(self.config.min_experts_per_layer, int(np.round(total_exp * target_ratio)))
            layer_budgets[l_idx] = target_k

        return layer_budgets

    def compute_specificity_scores(
        self,
        routing_weights: np.ndarray,
        active_threshold: float = 0.05
    ) -> np.ndarray:
        """
        Tính điểm Độ Đặc Thù Độc Quyền (Specificity Score) theo nguyên lý TF-IDF MoE:
        
        Specificity(e) = max_{t in Boundary} [ Exclusivity(e, t) * log(N_total / (Count(t | g_e(t) > tau) + 1)) ]
        
        - Exclusivity(e, t) = g_e(t) / (sum_k g_k(t) + eps): Đo độ độc quyền tại token hiểm hóc.
        - IAF(e) = log(N_total / (Count + 1)): Phạt expert đại trà, thưởng nặng expert ngách hiếm gặp.
        """
        # routing_weights: shape (N_tokens, Num_Experts)
        if routing_weights.ndim != 2 or routing_weights.shape[0] == 0:
            return np.zeros(routing_weights.shape[-1] if routing_weights.ndim > 0 else 1)

        n_tokens, num_exp = routing_weights.shape
        eps = 1e-8

        # 1. Exclusivity: Tỷ trọng độc quyền của expert e tại mỗi token t
        token_sum = np.sum(routing_weights, axis=-1, keepdims=True) + eps
        exclusivity = routing_weights / token_sum  # (N_tokens, Num_Experts)

        # 2. Inverse Activation Frequency (IAF): Nghịch đảo tần suất kích hoạt
        active_counts = np.sum(routing_weights > active_threshold, axis=0)  # (Num_Experts,)
        iaf = np.log((n_tokens + 1.0) / (active_counts + 1.0))  # (Num_Experts,)

        # 3. Specificity: Cực đại tích số Exclusivity * IAF trên toàn bộ tập token biên
        token_specificities = exclusivity * iaf[np.newaxis, :]  # (N_tokens, Num_Experts)
        specificity_scores = np.max(token_specificities, axis=0)  # (Num_Experts,)

        return specificity_scores

    def select_subnet(self, profile_data: Dict[str, Any]) -> Dict[int, List[int]]:
        """
        Thực hiện thuật toán Specificity-Pinned Entangled Bundle Selection:
        1. Khóa chặt (Auto-PIN) các chuyên gia ngách hiểm hóc có Specificity >= Theta_critical.
        2. Gom cụm các chuyên gia đồng kích hoạt (Co-Activation Entangled Bundles) bằng Combined Score.
        3. Điền đầy ngân sách 128 experts đồng nhất theo chuẩn HuggingFace.
        """
        layers_data = profile_data["layers"]
        num_layers = profile_data["num_layers"]
        
        num_experts_map = {int(l_idx): data["num_experts"] for l_idx, data in layers_data.items()}
        budgets = self.compute_layer_expert_budgets(num_layers, num_experts_map)
        
        selected_experts_per_layer: Dict[int, List[int]] = {}
        
        for l_idx_str, data in layers_data.items():
            l_idx = int(l_idx_str)
            target_k = budgets[l_idx]
            num_exp = data["num_experts"]
            
            importance = np.array(data["importance_scores"])  # (Num_Experts,)
            coact = np.array(data["coactivation_matrix"])     # (Num_Experts, Num_Experts)
            
            # Tính hoặc lấy Specificity Scores
            if "specificity_scores" in data:
                specificity = np.array(data["specificity_scores"])
            elif "routing_weights" in data:
                specificity = self.compute_specificity_scores(
                    np.array(data["routing_weights"]),
                    active_threshold=self.config.active_gate_threshold
                )
            else:
                # Ước lượng Specificity từ ma trận Co-activation và Importance
                mean_coact = np.mean(coact, axis=1) + 1e-8
                specificity = importance / mean_coact
                specificity = (specificity - specificity.min()) / (specificity.max() - specificity.min() + 1e-8) * 3.0

            selected: Set[int] = set()

            # --- BƯỚC 1: CRITICAL SPECIFICITY PINNING RULE ---
            # Khóa chặt tuyệt đối các chuyên gia ngách hiểm hóc (Anti-REAP Frequency Bias)
            if self.config.enable_specificity_pinning:
                pinned_candidates = np.where(specificity >= self.config.critical_specificity_threshold)[0]
                # Sắp xếp candidate theo specificity giảm dần
                pinned_candidates = sorted(pinned_candidates, key=lambda e: specificity[e], reverse=True)
                for exp in pinned_candidates:
                    if len(selected) >= target_k:
                        break
                    selected.add(int(exp))

            # --- BƯỚC 2: COMBINED SCORE RANKING ---
            # Kết hợp cân bằng giữa Tần suất (Frequency) và Độ Đặc thù (Specificity)
            beta = self.config.specificity_weight
            norm_imp = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            norm_spec = (specificity - specificity.min()) / (specificity.max() - specificity.min() + 1e-8)
            combined_scores = (1.0 - beta) * norm_imp + beta * norm_spec

            sorted_indices = np.argsort(-combined_scores).tolist()

            # --- BƯỚC 3: CO-ACTIVATION ENTANGLED BUNDLE SELECTION ---
            for primary_expert in sorted_indices:
                if len(selected) >= target_k:
                    break
                    
                if primary_expert not in selected:
                    selected.add(primary_expert)
                    
                    # Tìm các chuyên gia đồng kích hoạt mạnh
                    correlations = coact[primary_expert]
                    entangled_candidates = np.where(correlations >= self.config.coactivation_threshold)[0]
                    
                    entangled_candidates = sorted(
                        entangled_candidates,
                        key=lambda e: combined_scores[e],
                        reverse=True
                    )
                    
                    for cand in entangled_candidates:
                        if len(selected) >= target_k:
                            break
                        selected.add(int(cand))

            # --- BƯỚC 4: FILL REMAINING SLOTS ---
            for exp in sorted_indices:
                if len(selected) >= target_k:
                    break
                selected.add(int(exp))

            selected_list = sorted(list(selected))
            selected_experts_per_layer[l_idx] = selected_list

        return selected_experts_per_layer
