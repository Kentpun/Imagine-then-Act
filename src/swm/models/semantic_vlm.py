from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from swm.data.prompts import build_prompt


def build_model_and_processor(cfg: dict[str, Any], *, for_training: bool = True):
    model_cfg = cfg["model"]
    model_id = model_cfg["model_id"]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and "max_pixels" in model_cfg:
        processor.image_processor.max_pixels = int(model_cfg["max_pixels"])

    quant_config = None
    if model_cfg.get("load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if model_cfg.get("bf16", True)
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=dtype if quant_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if for_training:
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        if cfg["train"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        if model_cfg.get("freeze_vision", False):
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        lora_cfg = model_cfg["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg["target_modules"]),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor


def load_finetuned(cfg: dict[str, Any], checkpoint: str):
    """Load base model + LoRA adapter for inference."""
    model, processor = build_model_and_processor(cfg, for_training=False)
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_outcome(
    model,
    processor,
    image: Image.Image,
    task: str,
    action: list[float],
    *,
    prompt: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    prompt = prompt if prompt is not None else build_prompt(task, action)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
