"""
Fast Router Calibration / Healing Engine for pruned MoE models.
"""

from typing import List, Optional, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from moe_slice.config import SliceConfig


class RouterCalibrator:
    """
    Cơ chế Router Healing siêu tốc:
    - Đóng băng 100% weights của Attention, Norms, Embeddings và Expert FFNs.
    - Chỉ cho phép cập nhật trọng số của các Router/Gate layers.
    - Tối ưu hóa lại phân phối Softmax trên tập chuyên gia còn lại trong vài phút.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: SliceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def calibrate(self, calibration_prompts: List[str]) -> nn.Module:
        """
        Thực hiện quá trình hiệu chuẩn router weights.
        """
        print("[*] Bắt đầu quá trình Router Healing (Hiệu chuẩn Router)...")
        
        # 1. Freeze toàn bộ mô hình
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Mở gradient duy nhất cho các Gate/Router parameters
        router_params = []
        
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            transformer_layers = self.model.transformer.layers
        elif hasattr(self.model, "layers"):
            transformer_layers = self.model.layers

        for layer in transformer_layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                for p in layer.mlp.gate.parameters():
                    p.requires_grad = True
                    router_params.append(p)
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "gate"):
                for p in layer.block_sparse_moe.gate.parameters():
                    p.requires_grad = True
                    router_params.append(p)

        print(f"[*] Tổng số tham số Router cần hiệu chuẩn: {sum(p.numel() for p in router_params):,} parameters.")
        
        if len(router_params) == 0:
            print("[!] Cảnh báo: Không tìm thấy tham số router nào để calibrate. Bỏ qua bước này.")
            return self.model

        optimizer = AdamW(router_params, lr=self.config.calibration_lr, weight_decay=0.01)
        self.model.train()

        total_steps = min(self.config.calibration_steps, len(calibration_prompts))
        
        for step in range(total_steps):
            prompt = calibration_prompts[step % len(calibration_prompts)]
            device = getattr(self.model, "device", self.config.device)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(device)
            
            inputs["labels"] = inputs["input_ids"].clone()

            optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
                optimizer.step()

            if (step + 1) % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                loss_val = loss.item() if loss is not None else 0.0
                print(f"    [Step {step + 1}/{total_steps}] Calibration Loss: {loss_val:.4f}")

        self.model.eval()
        print("[✓] Hoàn tất Router Healing thành công!")
        return self.model
