#!/usr/bin/env python3
"""Prepare multi-task RoboCasa HF LeRobot slices for offline Layer A OOD eval.

Downloads particular task datasets (not a random 1% of RoboCasa365), extracts
frames, and writes semantic transitions compatible with swm.eval.evaluate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from swm.data.labeling import LabelConfig, build_outcome_string, label_transition
from swm.utils.config import load_config
from swm.utils.io import ensure_dir, write_jsonl
from swm.utils.seed import set_seed

# Particular published tasks (plan lock).
DEFAULT_REPOS = [
    "pepijn223/atomic_4",
    "pepijn223/robocasa_PickPlaceCounterToCabinet",
    "pepijn223/robocasa_PrepareCoffee",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/robocasa_ood.yaml")
    p.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="HF dataset repo ids (default: atomic_4 + 2 extras).",
    )
    p.add_argument("--max_per_task", type=int, default=96)
    p.add_argument("--stride", type=int, default=5)
    return p.parse_args()


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, list) and v:
        return _as_str(v[0])
    s = str(v).strip()
    if not s or s.isdigit():
        return None
    return s


def _task_name(ex: dict[str, Any], repo: str) -> str:
    for key in (
        "task",
        "language_instruction",
        "annotation.human.task_name",
        "annotation.human.task_description",
        "tasks",
    ):
        s = _as_str(ex.get(key))
        if s:
            return s
    return repo.split("robocasa_")[-1].split("/")[-1]


def _get_image(ex: dict[str, Any]) -> Image.Image | None:
    for key in (
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
        "observation.images.image",
        "image",
    ):
        if key not in ex or ex[key] is None:
            continue
        img = ex[key]
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, dict) and "path" in img:
            return Image.open(img["path"]).convert("RGB")
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (
                (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                if arr.max() <= 1.0
                else np.clip(arr, 0, 255).astype(np.uint8)
            )
        return Image.fromarray(arr)
    return None


def _load_lerobot(repo: str, root: Path, *, episodes: list[int] | None = None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    root.mkdir(parents=True, exist_ok=True)
    # Untagged HF repos (PickPlace/PrepareCoffee) have no vX.Y tags; LeRobot's
    # get_safe_version() crashes. Force branch revision. Also force PyAV because
    # SuperPOD torchcodec/FFmpeg .so links are broken.
    print(f"Loading {repo} → {root} (revision=main, video_backend=pyav) ...")
    kwargs: dict[str, Any] = {
        "root": str(root),
        "revision": "main",
        "video_backend": "pyav",
        "download_videos": True,
    }
    if episodes is not None:
        kwargs["episodes"] = episodes
    return LeRobotDataset(repo, **kwargs)


def _episode_index_map(ds) -> dict[int, list[int]]:
    """Group frame indices by episode without decoding video."""
    ep_to_idx: dict[int, list[int]] = {}
    hf = getattr(ds, "hf_dataset", None)
    if hf is not None and "episode_index" in hf.column_names:
        for i, ep in enumerate(hf["episode_index"]):
            ep_to_idx.setdefault(int(ep), []).append(i)
        return ep_to_idx
    for i in range(len(ds)):
        ex = ds[i]
        ep = int(ex.get("episode_index", ex.get("episode_id", i // 50)))
        ep_to_idx.setdefault(ep, []).append(i)
    return ep_to_idx


def materialize_repo(
    repo: str,
    *,
    raw_root: Path,
    frames_dir: Path,
    label_cfg: LabelConfig,
    max_per_task: int,
    stride: int,
    start_ep: int,
) -> list[dict]:
    root = raw_root / re.sub(r"[^a-zA-Z0-9_.-]+", "_", repo)
    # Cap episodes so we do not decode hundreds of thousands of frames.
    # Buffer is large enough for multi-task repos (atomic_4 has 7 skills).
    max_episodes = max(80, max_per_task)
    ds = _load_lerobot(repo, root, episodes=list(range(max_episodes)))
    ep_to_idx = _episode_index_map(ds)
    print(f"  loaded {len(ds)} frames across {len(ep_to_idx)} episodes")

    rows: list[dict] = []
    per_task_count: dict[str, int] = {}
    for ep_i, (ep, idxs) in enumerate(
        tqdm(sorted(ep_to_idx.items()), desc=f"frames:{repo.split('/')[-1]}")
    ):
        idxs = sorted(idxs)
        first = ds[idxs[0]]
        task = _task_name(first, repo)
        if per_task_count.get(task, 0) >= max_per_task:
            continue
        last = ds[idxs[-1]]
        episode_success = bool(
            last.get("next.success")
            or last.get("success")
            or last.get("next.done")
            or True
        )
        for local_t, i in enumerate(idxs[:-1]):
            if local_t % stride != 0:
                continue
            if per_task_count.get(task, 0) >= max_per_task:
                break
            ex = ds[i]
            ex_next = ds[idxs[min(local_t + 1, len(idxs) - 1)]]
            image = _get_image(ex)
            if image is None:
                continue
            action = ex.get("action")
            if action is None:
                continue
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            # Pad/truncate to 7 for LIBERO-style labeling
            a7 = np.zeros(7, dtype=np.float32)
            a7[: min(7, action.shape[0])] = action[:7]
            state = ex.get("observation.state", ex.get("state"))
            state_next = ex_next.get("observation.state", ex_next.get("state"))
            labels = label_transition(
                a7,
                state,
                state_next,
                frame_idx=local_t,
                episode_len=len(idxs),
                episode_success=episode_success,
                cfg=label_cfg,
            )
            global_ep = start_ep + int(ep)
            rel = f"{repo.split('/')[-1]}_ep{global_ep:05d}_t{local_t:04d}.jpg"
            out_path = frames_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                image.save(out_path, quality=90)
            rows.append(
                {
                    "episode_id": global_ep,
                    "frame_idx": int(local_t),
                    "task": task,
                    "task_name": task,
                    "source_repo": repo,
                    "action": a7.tolist(),
                    "outcome": build_outcome_string(labels),
                    "image_path": str(out_path),
                }
            )
            per_task_count[task] = per_task_count.get(task, 0) + 1
    print(f"  {repo}: tasks={per_task_count}")
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    repos = args.repos or cfg["data"].get("repos") or DEFAULT_REPOS
    processed = ensure_dir(cfg["paths"]["processed_dir"])
    frames = ensure_dir(cfg["paths"]["frames_dir"])
    raw = ensure_dir(cfg["paths"]["raw_dir"])
    label_cfg = LabelConfig(
        move_threshold=float(cfg["data"].get("move_threshold", 0.01)),
        lift_threshold=float(cfg["data"].get("lift_threshold", 0.01)),
        success_near_frames=int(cfg["data"].get("success_near_frames", 15)),
    )
    max_per_task = int(args.max_per_task or cfg["data"].get("max_per_task", 96))
    stride = int(args.stride or cfg["data"].get("stride", 5))

    all_rows: list[dict] = []
    ep_base = 0
    failed: list[str] = []
    for repo in repos:
        try:
            rows = materialize_repo(
                repo,
                raw_root=raw,
                frames_dir=frames,
                label_cfg=label_cfg,
                max_per_task=max_per_task,
                stride=stride,
                start_ep=ep_base,
            )
            all_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001 — keep going for other OOD tasks
            failed.append(repo)
            print(f"WARNING: skipping {repo}: {type(exc).__name__}: {exc}")
        ep_base += 10_000

    if not all_rows:
        raise RuntimeError(f"No RoboCasa OOD rows prepared; failed={failed}")

    # All OOD rows go to val.jsonl for evaluation; tiny train stub for tooling.
    n_val = write_jsonl(Path(processed) / "val.jsonl", all_rows)
    n_train = write_jsonl(Path(processed) / "train.jsonl", all_rows[: min(32, len(all_rows))])
    meta = {
        "repos": repos,
        "failed_repos": failed,
        "n_val": n_val,
        "n_train": n_train,
        "max_per_task": max_per_task,
        "tasks": sorted({r["task"] for r in all_rows}),
    }
    (Path(processed) / "ood_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {n_val} OOD val transitions; tasks={meta['tasks']}")


if __name__ == "__main__":
    main()
