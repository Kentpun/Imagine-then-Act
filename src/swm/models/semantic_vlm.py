from __future__ import annotations

import gc
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
    mps_train = bool(cfg.get("train", {}).get("mps_train", False))
    # MPS is happier with float16 for inference and bfloat16 for local training.
    if use_cuda and cfg["train"].get("bf16", True):
        dtype = torch.bfloat16
    elif use_mps and for_training and mps_train:
        dtype = torch.float16
    elif use_cuda or use_mps:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Training: avoid device_map="auto" (often hangs / mis-places grads with PEFT on Windows).
    # 4-bit needs a fixed map onto cuda:0. Full bf16/fp16: load with low_cpu_mem_usage + SDPA.
    attn_impl = model_cfg.get("attn_implementation", "sdpa")
    if quant_config is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": 0} if use_cuda else None,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        if use_cuda:
            model = model.to("cuda")
        elif use_mps and (not for_training or mps_train):
            model = model.to("mps")
        else:
            model = model.to("cpu")

    resume_from = cfg.get("train", {}).get("resume_from")
    if for_training and resume_from:
        if hasattr(model, "config"):
            model.config.use_cache = False
        # Continue an existing LoRA instead of creating a fresh adapter.
        model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
        if model_cfg.get("freeze_vision", False):
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False
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
    """Load base model + LoRA adapter for inference with minimal RAM/VRAM overhead."""
    model, processor = build_model_and_processor(cfg, for_training=False)
    model = PeftModel.from_pretrained(model, checkpoint)
    is_4bit = bool(cfg.get("model", {}).get("load_in_4bit", False))
    want_merge = bool(
        cfg.get("planner", {}).get("merge_lora", False)
        or cfg.get("model", {}).get("merge_lora", False)
    )
    if want_merge and not is_4bit:
        print("Merging LoRA adapters into base weights for faster inference...", flush=True)
        model = model.merge_and_unload()
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
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
    return_margins: bool = False,
) -> list[str] | tuple[list[str], list[float]]:
    """Batched greedy outcome generation across K candidate actions on one frame.

    When ``return_margins=True``, also returns ``P(progress=up) - P(progress=down)``
    extracted directly from the generation step logits (zero extra forward passes)
    so planners can break discrete argmax ties using calibrated world-model confidence.
    """
    if not actions:
        return ([], []) if return_margins else []
    prompt_list = (
        list(prompts)
        if prompts is not None
        else [build_prompt(task, a) for a in actions]
    )
    if len(actions) == 1 and not return_margins:
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
    tok_up = tok.encode("up", add_special_tokens=False)[0] if (tok is not None and return_margins) else -1
    tok_down = tok.encode("down", add_special_tokens=False)[0] if (tok is not None and return_margins) else -1
    tok_flat = tok.encode("flat", add_special_tokens=False)[0] if (tok is not None and return_margins) else -1
    prog_toks = {tok_up, tok_down, tok_flat}
    results: list[str] = []
    margins: list[float] = []
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
            in_len = inputs["input_ids"].shape[1]
            if return_margins and tok_up >= 0 and tok_down >= 0 and tok_flat >= 0:
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                seqs = out.sequences
                step_scores = out.scores
                trimmed = [row[in_len:] for row in seqs]
                for b_idx, row_toks in enumerate(trimmed):
                    m_val = 0.0
                    for t_idx, tid in enumerate(row_toks.tolist()):
                        if tid in prog_toks and t_idx < len(step_scores):
                            lg = step_scores[t_idx][b_idx, [tok_up, tok_down, tok_flat]].float()
                            pr_vec = torch.softmax(lg, dim=-1)
                            m_val = float(pr_vec[0].item() - pr_vec[1].item())
                            break
                    margins.append(m_val)
            else:
                seqs = model.generate(
                    **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
                )
                trimmed = [row[in_len:] for row in seqs]
                margins.extend([0.0] * len(trimmed))
            decoded = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.extend(s.strip() for s in decoded)
        return (results, margins) if return_margins else results
    except Exception:
        # Fallback to exact single-sample decode if processor/device rejects batching
        single = [
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
        return (single, [0.0] * len(single)) if return_margins else single
    finally:
        if tok is not None:
            tok.padding_side = old_side


@torch.inference_mode()
def generate_microduck_outcomes_fast(
    model,
    processor,
    image: Image.Image,
    task: str,
    actions: list[list[float]],
    *,
    prompts: list[str] | None = None,
    system: str | None = None,
    max_new_tokens: int = 40,
    batch_size: int = 8,
) -> tuple[list[str], list[float]]:
    """Shared-image KV-cache + 6-jump structured decoder for real-time MicroDuck control.

    Runs the Vision Transformer + shared task prefix ONCE (batch=1), expands the KV cache
    across all K candidate actions, and jumps across fixed schema delimiters in 6 tiny
    cached steps instead of 37 full autoregressive decode passes (~6x faster, ~4x less VRAM).
    Automatically falls back to ``generate_outcomes_batch`` if anything differs in length.
    """
    if not actions:
        return [], []
    prompt_list = (
        list(prompts)
        if prompts is not None
        else [build_prompt(task, a) for a in actions]
    )
    tok = getattr(processor, "tokenizer", None)
    if tok is None or len(actions) > int(batch_size):
        res = generate_outcomes_batch(
            model,
            processor,
            image,
            task,
            actions,
            prompts=prompt_list,
            system=system,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            return_margins=True,
        )
        assert isinstance(res, tuple)
        return res

    try:
        full_texts: list[str] = []
        for pr in prompt_list:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": pr},
                    ],
                }
            )
            full_texts.append(
                processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        # Split right at '\nCandidate action' so the entire <image> + Task header runs ONCE (batch=1).
        marker = "\nCandidate action"
        split_idx = full_texts[0].find(marker)
        if split_idx <= 0 or not all(t[: split_idx + 1] == full_texts[0][: split_idx + 1] for t in full_texts):
            raise RuntimeError("Prompts do not share common vision+task prefix")
        common_prefix = full_texts[0][: split_idx + 1]  # includes trailing '\n'

        suf_ids = [
            tok.encode(t[split_idx + 1 :] + "moved=", add_special_tokens=False)
            for t in full_texts
        ]
        if len({len(s) for s in suf_ids}) != 1:
            raise RuntimeError("Action suffixes differ in token count")

        def _tid(word: str) -> int:
            ids = tok.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Expected single token for {word!r}, got {ids}")
            return ids[0]

        id_yes, id_no = _tid("yes"), _tid("no")
        id_hold, id_close = _tid("hold"), _tid("close")
        id_up, id_down, id_flat = _tid("up"), _tid("down"), _tid("flat")
        id_walk, id_kick = _tid("walk"), _tid("kick")

        j_grip = tok.encode("; lifted=no; gripper=", add_special_tokens=False)
        j_prog = tok.encode("; progress=", add_special_tokens=False)
        j_succ = tok.encode("; success_near=", add_special_tokens=False)
        j_ball = tok.encode("; ball_in_zone=", add_special_tokens=False)
        j_face = tok.encode("; facing_goal=", add_special_tokens=False)
        j_skill = tok.encode("; skill=", add_special_tokens=False)

        device = get_model_device(model)
        k = len(full_texts)

        # 1. Single ViT + task prefix forward pass (batch_size = 1)
        pref_inputs = processor(
            text=[common_prefix], images=[image], return_tensors="pt"
        ).to(device)
        pref_out = model(**pref_inputs, use_cache=True)
        past = pref_out.past_key_values
        if k > 1:
            past.batch_repeat_interleave(k)

        def _pick(logits_last: torch.Tensor, opts: list[tuple[int, str]]) -> tuple[list[int], list[str]]:
            cand_ids = [o[0] for o in opts]
            sub = logits_last[:, cand_ids]
            best_idx = sub.argmax(dim=-1).tolist()
            return [opts[i][0] for i in best_idx], [opts[i][1] for i in best_idx]

        # 2. Per-candidate action suffix + 'moved='
        step_in = torch.tensor(suf_ids, dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        mv_ids, mv_s = _pick(out.logits[:, -1, :], [(id_yes, "yes"), (id_no, "no")])

        # 3. '; lifted=no; gripper='
        step_in = torch.tensor([[t] + j_grip for t in mv_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        gr_ids, gr_s = _pick(out.logits[:, -1, :], [(id_hold, "hold"), (id_close, "close")])

        # 4. '; progress=' (+ exact calibrated P(up) - P(down) tie-breaker)
        step_in = torch.tensor([[t] + j_prog for t in gr_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        pr_logits = out.logits[:, -1, [id_up, id_down, id_flat]].float()
        pr_probs = torch.softmax(pr_logits, dim=-1)
        margins = (pr_probs[:, 0] - pr_probs[:, 1]).tolist()
        pr_ids, pr_s = _pick(
            out.logits[:, -1, :], [(id_up, "up"), (id_down, "down"), (id_flat, "flat")]
        )

        # 5. '; success_near='
        step_in = torch.tensor([[t] + j_succ for t in pr_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        sn_ids, sn_s = _pick(out.logits[:, -1, :], [(id_yes, "yes"), (id_no, "no")])

        # 6. '; ball_in_zone='
        step_in = torch.tensor([[t] + j_ball for t in sn_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        bz_ids, bz_s = _pick(out.logits[:, -1, :], [(id_yes, "yes"), (id_no, "no")])

        # 7. '; facing_goal='
        step_in = torch.tensor([[t] + j_face for t in bz_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        fg_ids, fg_s = _pick(out.logits[:, -1, :], [(id_yes, "yes"), (id_no, "no")])

        # 8. '; skill=' (final jump — release cache right after)
        step_in = torch.tensor([[t] + j_skill for t in fg_ids], dtype=torch.long, device=device)
        out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        _, sk_s = _pick(
            out.logits[:, -1, :], [(id_walk, "walk"), (id_hold, "hold"), (id_kick, "kick")]
        )

        del past, pref_out, out
        outcomes = [
            f"moved={mv_s[i]}; lifted=no; gripper={gr_s[i]}; progress={pr_s[i]}; "
            f"success_near={sn_s[i]}; ball_in_zone={bz_s[i]}; "
            f"facing_goal={fg_s[i]}; skill={sk_s[i]}"
            for i in range(k)
        ]
        return outcomes, [float(m) for m in margins]
    except Exception:
        res = generate_outcomes_batch(
            model,
            processor,
            image,
            task,
            actions,
            prompts=prompt_list,
            system=system,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            return_margins=True,
        )
        assert isinstance(res, tuple)
        return res