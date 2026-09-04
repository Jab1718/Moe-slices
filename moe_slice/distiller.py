"""
Router Gate DoRA Distillation & Calibration Engine.
Calibrates routing probability distribution on sliced subnets to eliminate
boundary token jitter and indentation errors.
"""

import os
import time
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


class DoRADistiller:
    def __init__(
        self,
        model_path: str,
        rank: int = 16,
        alpha: int = 32,
        layers_to_tune: list = None
    ):
        self.model_path = model_path
        self.rank = rank
        self.alpha = alpha
        self.layers_to_tune = layers_to_tune or list(range(15, 48))

    def setup_model_and_adapter(self, device_map: dict = None):
        """Loads model and unlocks Router Gate parameters alongside DoRA projections."""
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            use_dora=True,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=self.layers_to_tune,
            lora_dropout=0.05,
            bias="none"
        )
        peft_model = get_peft_model(model, lora_config)

        # Unlock router gate weights on specified layers
        router_params = 0
        for name, param in peft_model.named_parameters():
            if "gate.weight" in name and any(f"layers.{l}." in name for l in self.layers_to_tune):
                param.requires_grad = True
                router_params += param.numel()

        print(f"[✓] Initialized DoRA (r={self.rank}, a={self.alpha}) and unlocked {router_params:,} Router Gate parameters.")
        return peft_model
