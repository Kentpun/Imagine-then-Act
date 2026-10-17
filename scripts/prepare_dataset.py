#!/usr/bin/env python3
"""Build labeled semantic transitions from LIBERO (or a synthetic smoke set)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from swm.data.labeling import LabelConfig, build_outcome_string, label_transition
from swm.utils.config import load_config
from swm.utils.io import ensure_dir, write_jsonl
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Generate a tiny synthetic dataset to validate the training loop.",
    )
    return p.parse_args()


def _save_frame(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


def build_smoke(cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    frames_dir = ensure_dir(cfg["paths"]["frames_dir"])
    label_cfg = LabelConfig(
        move_threshold=cfg["data"]["move_threshold"],
        lift_threshold=cfg["data"]["lift_threshold"],
        success_near_frames=cfg["data"]["success_near_frames"],
    )
    rows: list[dict] = []
    tasks = [
        "pick up the red cup and place it on the plate",
        "push the bowl to the left",
        "open the drawer",
    ]
    for ep in range(12):
        task = tasks[ep % len(tasks)]
        success = ep % 3 != 0
        T = 20
        state = np.zeros(8, dtype=np.float32)
        for t in range(0, T - 1, cfg["data"]["stride"]):
            action = np.array(
                [
                    random.uniform(-0.05, 0.05),
                    random.uniform(-0.05, 0.05),
                    random.uniform(-0.02, 0.03),
                    0.0,
                    0.0,
                    0.0,
                    float(random.random() > 0.4),
                ],
                dtype=np.float32,
            )
            state_next = state.copy()
            state_next[:3] += action[:3]
            state_next[7] = action[6]
            labels = label_transition(
                action,
                state,
                state_next,
                frame_idx=t,
                episode_len=T,
                episode_success=success,
                cfg=label_cfg,
            )
            # Synthetic RGB
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
            arr[:, :, 0] = int(80 + 40 * (ep % 4))
            arr[:, :, 1] = int(60 + t * 5)
            arr[:, :, 2] = int(100 + 20 * action[6])
            img = Image.fromarray(arr)
            rel = f"smoke_ep{ep:03d}_t{t:04d}.jpg"
            _save_frame(img, Path(frames_dir) / rel)
            rows.append(
                {
                    "episode_id": ep,
                    "frame_idx": t,
                    "task": task,
                    "action": action.tolist(),
                    "outcome": build_outcome_string(labels),
                    "image_path": str(Path(frames_dir) / rel),
                    "split_hint": "train" if ep < 10 else "val",
                }
            )
            state = state_next
    train = [r for r in rows if r["split_hint"] == "train"]
    val = [r for r in rows if r["split_hint"] == "val"]
    return train, val


def _pick_image_key(example: dict[str, Any], candidates: list[str]) -> str | None:
    for k in candidates:
        if k in example and example[k] is not None:
            return k
    # fuzzy
    for k in example:
        if "image" in k.lower():
            return k
    return None


def build_from_lerobot(cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    repo = cfg["data"]["repo_id"]
    root = cfg["paths"]["raw_dir"]
    ds = LeRobotDataset(repo, root=root)
    return _materialize_rows(cfg, ds)


def build_from_hf(cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Best-effort loader via datasets + local raw dir."""
    from datasets import load_dataset

    repo = cfg["data"]["repo_id"]
    # Streaming can be slow for random access; prefer local snapshot if present.
    local = Path(cfg["paths"]["raw_dir"])
    if (local / "meta").exists() or (local / "data").exists():
        try:
            return build_from_lerobot(cfg)
        except Exception as e:
            print(f"LeRobot local load failed ({e}); falling back to HF datasets.")

    ds = load_dataset(repo, split="train")
    return _materialize_rows(cfg, ds)


def _materialize_rows(cfg: dict[str, Any], ds) -> tuple[list[dict], list[dict]]:
    frames_dir = ensure_dir(cfg["paths"]["frames_dir"])
    label_cfg = LabelConfig(
        move_threshold=cfg["data"]["move_threshold"],
        lift_threshold=cfg["data"]["lift_threshold"],
        success_near_frames=cfg["data"]["success_near_frames"],
    )
    candidates = cfg["data"]["image_key_candidates"]
    max_episodes = cfg["data"].get("max_episodes")
    stride = int(cfg["data"]["stride"])

    # Group by episode if possible
    episode_to_indices: dict[Any, list[int]] = {}
    n = len(ds)
    print(f"Scanning {n} frames for episode boundaries...")
    for i in tqdm(range(n), desc="index"):
        ex = ds[i]
        ep = ex.get("episode_index", ex.get("episode_id", i // 100))
        if max_episodes is not None and int(ep) >= int(max_episodes):
            continue
        episode_to_indices.setdefault(int(ep), []).append(i)

    ep_ids = sorted(episode_to_indices.keys())
    random.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * float(cfg["data"]["val_ratio"])))
    val_eps = set(ep_ids[:n_val])

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for ep in tqdm(ep_ids, desc="episodes"):
        idxs = episode_to_indices[ep]
        idxs = sorted(idxs)
        # Infer task / success
        first = ds[idxs[0]]
        task = (
            first.get("task")
            or first.get("language_instruction")
            or first.get("tasks")
            or "complete the manipulation task"
        )
        if isinstance(task, list):
            task = task[0] if task else "complete the manipulation task"

        # Heuristic success: many LIBERO exports mark last reward / success
        last = ds[idxs[-1]]
        episode_success = bool(
            last.get("next.success")
            or last.get("success")
            or last.get("episode_success")
            or True
        )

        for local_t, i in enumerate(idxs[:-1]):
            if local_t % stride != 0:
                continue
            ex = ds[i]
            ex_next = ds[idxs[min(local_t + 1, len(idxs) - 1)]]
            img_key = _pick_image_key(ex, candidates)
            if img_key is None:
                raise KeyError(
                    f"No image key found. Available keys: {list(ex.keys())[:40]}"
                )
            img = ex[img_key]
            if isinstance(img, dict) and "path" in img:
                image = Image.open(img["path"]).convert("RGB")
            elif isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                # numpy / torch chw or hwc
                arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                    if arr.max() <= 1:
                        arr = (arr * 255).astype(np.uint8)
                image = Image.fromarray(arr)

            action = ex.get("action")
            if action is None:
                continue
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            state = ex.get("observation.state", ex.get("state"))
            state_next = ex_next.get("observation.state", ex_next.get("state"))
            labels = label_transition(
                action,
                state,
                state_next,
                frame_idx=local_t,
                episode_len=len(idxs),
                episode_success=episode_success,
                cfg=label_cfg,
            )
            rel = f"ep{ep:05d}_t{local_t:04d}.jpg"
            out_path = Path(frames_dir) / rel
            _save_frame(image, out_path)
            row = {
                "episode_id": int(ep),
                "frame_idx": int(local_t),
                "task": str(task),
                "action": action.tolist(),
                "outcome": build_outcome_string(labels),
                "image_path": str(out_path),
            }
            if ep in val_eps:
                val_rows.append(row)
            else:
                train_rows.append(row)

    return train_rows, val_rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    processed = ensure_dir(cfg["paths"]["processed_dir"])

    if args.smoke_test or Path(cfg["paths"]["raw_dir"], "SMOKE_TEST").exists():
        print("Building synthetic smoke-test dataset...")
        train_rows, val_rows = build_smoke(cfg)
    else:
        backend = cfg["data"].get("backend", "hf")
        print(f"Building dataset with backend={backend} ...")
        if backend == "lerobot":
            train_rows, val_rows = build_from_lerobot(cfg)
        else:
            train_rows, val_rows = build_from_hf(cfg)

    n_train = write_jsonl(Path(processed) / "train.jsonl", train_rows)
    n_val = write_jsonl(Path(processed) / "val.jsonl", val_rows)
    print(f"Wrote {n_train} train / {n_val} val transitions to {processed}")


if __name__ == "__main__":
    main()
