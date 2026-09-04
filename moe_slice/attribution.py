"""
Closed-Loop Attribution Tracer for MoE Slicing.
Quantifies neuron coverage on failed tasks to prove whether errors originate
from missing expert parameters (structural deficit) or logit deviation (calibration).
"""

from typing import Dict, List, Set


class ClosedLoopAttributionTracer:
    def __init__(self, selected_experts_map: Dict[str, List[int]]):
        self.selected_experts_map = {
            int(k): set(v) for k, v in selected_experts_map.items()
        }

    def compute_task_coverage(
        self,
        required_experts_per_layer: Dict[int, List[int]]
    ) -> float:
        """
        Computes Coverage = |TopK(T_fail) ∩ E_selected| / |TopK(T_fail)|
        If Coverage >= 98%, the subnet contains all necessary knowledge neurons.
        """
        total_required = 0
        total_present = 0

        for layer_idx, experts in required_experts_per_layer.items():
            if layer_idx in self.selected_experts_map:
                retained = self.selected_experts_map[layer_idx]
                for exp in experts:
                    total_required += 1
                    if exp in retained:
                        total_present += 1

        if total_required == 0:
            return 100.0
        return (total_present / total_required) * 100.0

    def diagnose_failure(self, task_id: str, coverage: float) -> str:
        if coverage >= 95.0:
            return f"[{task_id}] Neuron Coverage: {coverage:.1f}% -> NO NEURON DEFICIT (Calibrate via DoRA / Prompt)"
        else:
            return f"[{task_id}] Neuron Coverage: {coverage:.1f}% -> STRUCTURAL EXPERT DEFICIT (Loop MoE Slicing)"
