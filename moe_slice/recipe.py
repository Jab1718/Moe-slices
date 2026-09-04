"""
Recipe Parser and Execution Engine for moe-slice (YAML-based configuration).
"""

from typing import Dict, List, Optional, Any
import os
import yaml
from dataclasses import dataclass, asdict

from moe_slice.config import SliceConfig, LayerBudgetRule


@dataclass
class SliceRecipe:
    """
    Cấu trúc file công thức cắt lát MoE (YAML recipe).
    """
    model_name: str
    target_vram: str
    domain: str = "coding"
    output_dir: str = "./sliced_model"
    
    # Calibration config
    calibration_dataset: str = "builtin"  # 'builtin', 'custom', hoặc HF dataset ID
    custom_dataset_path: Optional[str] = None
    calibration_samples: int = 50
    
    # Retention strategy
    retention_ratio: float = 0.40  # Giữ 40% experts (24/60 experts)
    preserve_shared_experts: bool = True
    coactivation_threshold: float = 0.35
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SliceRecipe":
        """
        Đọc cấu hình từ file YAML.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file recipe tại: {yaml_path}")
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            model_name=data.get("model_name", "Qwen/Qwen1.5-MoE-A2.7B-Chat"),
            target_vram=str(data.get("target_vram", "24GB")),
            domain=data.get("domain", "coding"),
            output_dir=data.get("output_dir", "./sliced_model"),
            calibration_dataset=data.get("calibration", {}).get("dataset", "builtin"),
            custom_dataset_path=data.get("calibration", {}).get("file_path", None),
            calibration_samples=data.get("calibration", {}).get("samples", 50),
            retention_ratio=data.get("retention", {}).get("ratio", 0.40),
            preserve_shared_experts=data.get("retention", {}).get("preserve_shared_experts", True),
            coactivation_threshold=data.get("retention", {}).get("coactivation_threshold", 0.35)
        )

    def to_slice_config(self) -> SliceConfig:
        """
        Chuyển đổi sang SliceConfig nội bộ.
        """
        vram_num = float(''.join(c for c in self.target_vram if c.isdigit() or c == '.'))
        
        return SliceConfig(
            model_name_or_path=self.model_name,
            target_vram_gb=vram_num,
            domain=self.domain,
            output_dir=self.output_dir,
            preserve_shared_experts=self.preserve_shared_experts,
            coactivation_threshold=self.coactivation_threshold,
            layer_budget=LayerBudgetRule(
                early_layers_ratio=self.retention_ratio,
                middle_layers_ratio=self.retention_ratio,
                late_layers_ratio=self.retention_ratio
            )
        )
