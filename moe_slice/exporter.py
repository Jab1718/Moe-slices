"""
Hugging Face & Safetensors Exporter for pruned MoE models.
"""

from typing import Dict, List, Optional, Any
import os
import json
import torch
import torch.nn as nn
from moe_slice.config import SliceConfig


class MoEModelExporter:
    """
    Module xuất mô hình MoE sau khi cắt thành định dạng HuggingFace / Safetensors chuẩn:
    - Tương thích 100% với vLLM, Ollama, SGLang, transformers.
    - Cập nhật chính xác `config.json` với số lượng experts mới.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: SliceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def export(self, output_dir: Optional[str] = None, selected_experts_per_layer: Optional[Dict[int, List[int]]] = None):
        """
        Lưu toàn bộ weights, tokenizer và config đã cập nhật ra ổ đĩa.
        """
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[*] Bắt đầu xuất mô hình ra thư mục: {save_dir}")

        # 1. Cập nhật Model Config
        hf_config = self.model.config
        
        if selected_experts_per_layer is not None and len(selected_experts_per_layer) > 0:
            first_layer_num = len(list(selected_experts_per_layer.values())[0])
            hf_config.num_routed_experts = first_layer_num
            hf_config.num_experts = first_layer_num
            if hasattr(hf_config, "num_local_experts"):
                hf_config.num_local_experts = first_layer_num
                
            # Lưu metadata chi tiết về việc cắt tỉa
            metadata = {
                "moe_slice_version": "0.1.0",
                "domain": self.config.domain,
                "target_vram_gb": self.config.target_vram_gb,
                "selected_experts_per_layer": {str(k): v for k, v in selected_experts_per_layer.items()}
            }
            with open(os.path.join(save_dir, "moe_slice_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        # 2. Lưu weights dạng safetensors
        self.model.save_pretrained(save_dir, safe_serialization=True)
        
        # 3. Lưu tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
            
        print(f"[✓] Xuất mô hình thành công tại: {save_dir}")
        print(f"[✓] Mô hình có thể được tải bằng: AutoModelForCausalLM.from_pretrained('{save_dir}')")
