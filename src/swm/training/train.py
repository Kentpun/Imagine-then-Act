from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from swm.data.dataset import SemanticTransitionDataset, make_collate_fn
from swm.models.semantic_vlm import build_model_and_processor, get_model_device
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train semantic world model (LoRA)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def _dataloader_workers(cfg: dict) -> int:
    """Windows spawn + nested collate_fn often hits EOFError with workers > 0."""
    n = int(cfg["train"].get("dataloader_num_workers", 0))
    if sys.platform.startswith("win") and n > 0:
        print(
            f"Windows detected: forcing dataloader_num_workers=0 "
            f"(config had {n}) to avoid multiprocessing EOFError.",
            flush=True,
        )
        return 0
    return n


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    out_dir = ensure_dir(cfg["paths"]["output_dir"])

    train_path = processed / "train.jsonl"
    val_path = processed / "val.jsonl"
    if not train_path.is_file() or train_path.stat().st_size == 0:
        raise SystemExit(
            f"Missing/empty {train_path}. Run scripts/prepare_dataset.py first."
        )

    train_ds: torch.utils.data.Dataset = SemanticTransitionDataset(train_path, frames)
    if val_path.is_file() and val_path.stat().st_size > 0:
        val_ds: torch.utils.data.Dataset = SemanticTransitionDataset(val_path, frames)
    else:
        val_ds = train_ds
    if len(train_ds) == 0:
        raise SystemExit(
            "No training rows found. Run scripts/prepare_dataset.py first."
        )

    max_train_samples = cfg["train"].get("max_train_samples")
    if max_train_samples is not None:
        n = min(int(max_train_samples), len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        print(f"Capped train set to {n} samples (train.max_train_samples)", flush=True)

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val", flush=True)

    print(
        f"Loading model {cfg['model']['model_id']} "
        f"(first download/load can take several minutes)...",
        flush=True,
    )
    model, processor = build_model_and_processor(cfg, for_training=True)
    device = get_model_device(model)
    print(f"Model ready on device={device} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"mem allocated={torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )

    collate = make_collate_fn(processor, max_length=int(cfg["train"]["max_seq_length"]))
    num_workers = _dataloader_workers(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=False,
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
    print(
        f"Training: {cfg['train']['num_epochs']} epochs, "
        f"{steps_per_epoch} steps/epoch, ~{total_update_steps} optimizer updates.",
        flush=True,
    )

    use_bf16 = bool(cfg["train"].get("bf16", True)) and torch.cuda.is_available()
    use_fp16 = bool(cfg["train"].get("fp16", False)) and torch.cuda.is_available()
    # Prefer new torch.amp API when available.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    opt.zero_grad(set_to_none=True)
    history: list[dict] = []

    for epoch in range(int(cfg["train"]["num_epochs"])):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", mininterval=1.0)
        for step, batch in enumerate(pbar, start=1):
            if epoch == 0 and step == 1:
                print("Collated first batch; moving to GPU...", flush=True)
            batch = {k: v.to(device, non_blocking=False) for k, v in batch.items()}
            if epoch == 0 and step == 1:
                shapes = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
                print(f"Batch shapes: {shapes}", flush=True)
                print("Forward pass starting (may take several minutes once)...", flush=True)

            ctx = torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda" and (use_bf16 or use_fp16),
            )
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss / max(1, int(cfg["train"]["grad_accum"]))

            if epoch == 0 and step == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"Forward done. loss={float(loss.item()):.4f}; backward...", flush=True)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if epoch == 0 and step == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print("Backward done. Training loop is healthy.", flush=True)

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
                    val_loss = evaluate_loss(model, val_loader, device, use_bf16, use_fp16)
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
    print(f"Saved final checkpoint to {final_dir}", flush=True)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_bf16: bool, use_fp16: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=device.type == "cuda" and (use_bf16 or use_fp16),
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
