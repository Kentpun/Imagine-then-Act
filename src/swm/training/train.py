from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from swm.data.dataset import SemanticTransitionDataset, make_collate_fn
from swm.models.semantic_vlm import build_model_and_processor
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train semantic world model (LoRA)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    out_dir = ensure_dir(cfg["paths"]["output_dir"])

    train_ds = SemanticTransitionDataset(processed / "train.jsonl", frames)
    val_ds = SemanticTransitionDataset(processed / "val.jsonl", frames)
    if len(train_ds) == 0:
        raise SystemExit(
            "No training rows found. Run scripts/prepare_dataset.py first."
        )

    model, processor = build_model_and_processor(cfg, for_training=True)
    collate = make_collate_fn(processor, max_length=int(cfg["train"]["max_seq_length"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("dataloader_num_workers", 0)),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    steps_per_epoch = max(1, len(train_loader))
    total_update_steps = (
        steps_per_epoch
        * int(cfg["train"]["num_epochs"])
        // max(1, int(cfg["train"]["grad_accum"]))
    )
    warmup = int(total_update_steps * float(cfg["train"]["warmup_ratio"]))
    sched = get_cosine_schedule_with_warmup(opt, warmup, max(1, total_update_steps))

    use_bf16 = bool(cfg["train"].get("bf16", True)) and torch.cuda.is_available()
    use_fp16 = bool(cfg["train"].get("fp16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    opt.zero_grad(set_to_none=True)
    history: list[dict] = []

    for epoch in range(int(cfg["train"]["num_epochs"])):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            ctx = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=torch.cuda.is_available() and (use_bf16 or use_fp16),
            )
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss / max(1, int(cfg["train"]["grad_accum"]))

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.item())
            if step % int(cfg["train"]["grad_accum"]) == 0:
                if use_fp16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(cfg["train"]["logging_steps"]) == 0:
                    avg = running / int(cfg["train"]["logging_steps"])
                    pbar.set_postfix(loss=f"{avg:.4f}", step=global_step)
                    history.append({"step": global_step, "loss": avg, "epoch": epoch})
                    running = 0.0

                if global_step % int(cfg["train"]["save_steps"]) == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)

                if (
                    global_step % int(cfg["train"]["eval_steps"]) == 0
                    and len(val_ds) > 0
                ):
                    val_loss = evaluate_loss(model, val_loader, use_bf16, use_fp16)
                    history.append(
                        {"step": global_step, "val_loss": val_loss, "epoch": epoch}
                    )
                    model.train()

        # end-epoch save
        epoch_dir = out_dir / f"checkpoint-epoch-{epoch+1}"
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

    final_dir = out_dir / "checkpoint-final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved final checkpoint to {final_dir}")


@torch.no_grad()
def evaluate_loss(model, loader, use_bf16: bool, use_fp16: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=torch.cuda.is_available() and (use_bf16 or use_fp16),
        )
        with ctx:
            loss = model(**batch).loss
        total += float(loss.item())
        n += 1
        if n >= 50:
            break
    return total / max(1, n)


if __name__ == "__main__":
    main()
