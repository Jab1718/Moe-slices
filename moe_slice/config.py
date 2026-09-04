"""
Configuration dataclasses for the moe_slice framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class LayerBudgetRule:
    """
    Quy tắc phân bổ tỷ lệ chuyên gia theo độ sâu của tầng Transformer (Golden Ratio).
    """
    early_layers_ratio: float = 0.65    # Layers 0 -> N_early: Giữ 65% để bảo toàn cú pháp và token embeddings
    middle_layers_ratio: float = 0.40   # Layers N_early -> N_late: Tỷ lệ vàng 40% (24/60 experts) bao phủ toàn bộ DSA + Systems + Reasoning
    late_layers_ratio: float = 0.45     # Layers N_late -> End: Giữ 45% cho định dạng Markdown, Indentation, JSON
    early_boundary_fraction: float = 0.25  # 25% layers đầu
    late_boundary_fraction: float = 0.75   # 25% layers cuối


@dataclass
class SliceConfig:
    """
    Cấu hình tổng thể cho quá trình profiling, clustering và pruning MoE.
    """
    # Mục tiêu phần cứng & VRAM
    target_vram_gb: Optional[float] = 24.0  # Ví dụ 16.0, 24.0, 32.0 GB
    model_name_or_path: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    output_dir: str = "./sliced_model"
    
    # Chiến lược Domain & Bó Chuyên gia
    domain: str = "coding"  # 'coding', 'math', 'general', 'multilingual'
    coactivation_threshold: float = 0.50  # Ngưỡng tương quan tối thiểu để bó chuyên gia
    min_experts_per_layer: int = 4  # Số chuyên gia tối thiểu cần giữ lại trên mỗi tầng
    
    # Shared Experts Policy (Bắt buộc cho DeepSeek/Qwen)
    preserve_shared_experts: bool = True
    
    # Phân bổ ngân sách tầng
    layer_budget: LayerBudgetRule = field(default_factory=LayerBudgetRule)
    
    # Router Calibration (Healing)
    calibration_steps: int = 100
    calibration_lr: float = 5e-4
    calibration_batch_size: int = 2
    max_seq_length: int = 1024
    
    # Specificity & Niche Expert Retention (TF-IDF MoE Principle - Anti-REAP Frequency Bias)
    critical_specificity_threshold: float = 2.0   # Ngưỡng Theta_critical để tự động PIN expert
    specificity_weight: float = 0.5               # Trọng số beta (0.5 Frequency + 0.5 Specificity)
    active_gate_threshold: float = 0.05           # Ngưỡng tau_active tính kích hoạt token
    enable_specificity_pinning: bool = True       # Bật cơ chế khóa chặt chuyên gia ngách hiểm hóc
    
    # Hardware & Performance
    device: str = "cuda"
    dtype: str = "bfloat16"  # 'bfloat16', 'float16', 'float32'
