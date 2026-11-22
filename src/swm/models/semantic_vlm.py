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


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Resolve device for models with or without device_map."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_and_processor(cfg: dict[str, Any], *, for_training: bool = True):
    model_cfg = cfg["model"]
    model_id = model_cfg["model_id"]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and "max_pixels" in model_cfg:
        # Qwen VL: size.longest/shortest_edge == max/min pixel budget (not edge px).
        ip = processor.image_processor
        max_pixels = int(model_cfg["max_pixels"])
        min_pixels = int(model_cfg.get("min_pixels", max(256, max_pixels // 4)))
        if hasattr(ip, "size"):
            ip.size = {
                "longest_edge": max_pixels,
                "shortest_edge": min_pixels,
            }
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = min_pixels

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

    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    # MPS is happier with float16/float32 than bf16 for local Mac inference.
    if use_cuda and cfg["train"].get("bf16", True):
        dtype = torch.bfloat16
    elif use_cuda or use_mps:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Training: avoid device_map="auto" (often hangs / mis-places grads with PEFT on Windows).
    # 4-bit needs a fixed map onto cuda:0. Full bf16/fp16: load then .to(device).
    if quant_config is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": 0} if use_cuda else None,
            trust_remote_code=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if use_cuda:
            model = model.to("cuda")
        elif use_mps and not for_training:
            model = model.to("mps")
        else:
            model = model.to("cpu")

    resume_from = cfg.get("train", {}).get("resume_from")
    if for_training and resume_from:
        # Continue an existing LoRA instead of creating a fresh adapter.
        model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
        model.print_trainable_parameters()
        return model, processor

    if for_training:
        # Must disable KV cache before gradient checkpointing.
        if hasattr(model, "config"):
            model.config.use_cache = False
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        if cfg["train"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False

        if model_cfg.get("freeze_vision", False):
            n_frozen = 0
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False
                    n_frozen += 1
            print(
                f"Ablation: froze {n_frozen} vision parameters "
                f"(model.freeze_vision=true)",
                flush=True,
            )

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
    # Merge LoRA into base weights for faster Mac/CPU demos when requested.
    if cfg.get("planner", {}).get("merge_lora", False) or cfg.get("model", {}).get(
        "merge_lora", False
    ):
        print("Merging LoRA adapters into base weights for faster inference...", flush=True)
        model = model.merge_and_unload()
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_text(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    system: str | None = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """Free-form VLM generation (proposals / preference ranking)."""
    messages = []
    if system:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]}
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = get_model_device(model)
    inputs = inputs.to(device)
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False
    out = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


@torch.inference_mode()
def generate_outcome(
    model,
    processor,
    image: Image.Image,
    task: str,
    action: list[float],
    *,
    prompt: str | None = None,
    system: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    prompt = prompt if prompt is not None else build_prompt(task, action)
    return generate_text(
        model,
        processor,
        image,
        prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


@torch.inference_mode()
def generate_outcomes_batch(
    model,
    processor,
    image: Image.Image,
    task: str,
    actions: list[list[float]],
    *,
    prompts: list[str] | None = None,
    system: str | None = None,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Batched greedy outcome generation across K candidate actions on one frame."""
    if not actions:
        return []
    prompt_list = (
        list(prompts)
        if prompts is not None
        else [build_prompt(task, a) for a in actions]
    )
    if len(actions) == 1:
        return [
            generate_outcome(
                model,
                processor,
                image,
                task,
                actions[0],
                prompt=prompt_list[0],
                system=system,
                max_new_tokens=max_new_tokens,
            )
        ]

    device = get_model_device(model)
    tok = getattr(processor, "tokenizer", None)
    old_side = getattr(tok, "padding_side", "right") if tok is not None else "right"
    results: list[str] = []
    try:
        if tok is not None:
            tok.padding_side = "left"
        for start in range(0, len(prompt_list), max(1, int(batch_size))):
            sub_prompts = prompt_list[start : start + max(1, int(batch_size))]
            texts: list[str] = []
            imgs: list[Image.Image] = []
            for pr in sub_prompts:
                msgs = []
                if system:
                    msgs.append(
                        {"role": "system", "content": [{"type": "text", "text": system}]}
                    )
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": pr},
                        ],
                    }
                )
                texts.append(
                    processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                )
                imgs.append(image)
            inputs = processor(
                text=texts, images=imgs, padding=True, return_tensors="pt"
            ).to(device)
            out = model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
            )
            in_len = inputs["input_ids"].shape[1]
            trimmed = [row[in_len:] for row in out]
            decoded = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.extend(s.strip() for s in decoded)
        return results
    except Exception:
        # Fallback to exact single-sample decode if processor/device rejects batching
        return [
            generate_outcome(
                model,
                processor,
                image,
                task,
                a,
                prompt=pr,
                system=system,
                max_new_tokens=max_new_tokens,
            )
            for a, pr in zip(actions, prompt_list)
        ]
    finally:
        if tok is not None:
            tok.padding_side = old_side