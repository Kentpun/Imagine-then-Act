#!/usr/bin/env python3
"""Prepare robot-independent action-command rows from LeRobot datasets.

Each public trajectory is converted to:
  image + natural-language instruction -> one text action command

The stored answer contains no robot joint angles, MicroDuck velocities, MQTT
fields, or task-specific labels such as push_in.
"""

from __future__ import annotations

import argparse
import random
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from swm.data.action_commands import build_action_prompt, canonical_action_command
from swm.utils.config import load_config
from swm.utils.io import ensure_dir, write_jsonl
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/superpod_action.yaml")
    return parser.parse_args()


def _patch_lerobot_untagged_hub_datasets() -> None:
    """Some Hub LeRobot v3 datasets (e.g. nvidia Bridge) ship without version tags.

    LeRobot 0.4 then raises inside ``get_safe_version``. Fall back to ``main`` so
    episode-subset downloads still work.
    """
    try:
        from lerobot.datasets import utils as lerobot_utils
    except Exception:
        return
    if getattr(lerobot_utils.get_safe_version, "_swm_untagged_patch", False):
        return
    original = lerobot_utils.get_safe_version

    def get_safe_version(repo_id, version):  # noqa: ANN001
        try:
            return original(repo_id, version)
        except Exception as exc:
            print(
                f"Warning: no usable Hub version tags for {repo_id} ({exc}); "
                "using revision 'main'."
            )
            return "main"

    get_safe_version._swm_untagged_patch = True  # type: ignore[attr-defined]
    lerobot_utils.get_safe_version = get_safe_version


def _load_lerobot_dataset_class():
    _patch_lerobot_untagged_hub_datasets()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "LeRobot is required for the public action datasets. Install it in "
                "the SuperPOD venv with: pip install 'lerobot>=0.3'"
            ) from exc
    return LeRobotDataset


def _load_lerobot_metadata_class():
    _patch_lerobot_untagged_hub_datasets()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
        except Exception:
            return None
    return LeRobotDatasetMetadata


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"]).convert("RGB")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.size and float(np.max(arr)) <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0 or len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, num=count)
    return [values[int(round(p))] for p in positions]


def _episode_bounds(dataset: Any, episodes: list[int]) -> dict[int, tuple[int, int]]:
    """Map each selected episode id to a half-open [start, stop) index range.

    When LeRobot loads a subset with ``episodes=[...]``, ``meta.episodes`` still
    stores *global* frame indices into the full dataset. Indexing the subset
    with those globals raises IndexError. Prefer ranges scanned from the loaded
    HF table, which are always local to ``dataset[i]``.
    """
    wanted = set(episodes)
    bounds: dict[int, tuple[int, int]] = {}

    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and len(hf_dataset) > 0 and "episode_index" in hf_dataset.column_names:
        episode_ids = [int(v) for v in hf_dataset["episode_index"]]
        current: int | None = None
        start = 0
        for index, episode in enumerate(episode_ids):
            if current is None:
                current = episode
                start = index
            elif episode != current:
                if current in wanted:
                    bounds[current] = (start, index)
                current = episode
                start = index
        if current is not None and current in wanted:
            bounds[current] = (start, len(episode_ids))
        if bounds:
            return bounds

    meta_episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    records: list[dict[str, Any]] = []
    if isinstance(meta_episodes, dict) and "dataset_from_index" in meta_episodes:
        ep_ids = meta_episodes.get(
            "episode_index", list(range(len(meta_episodes["dataset_from_index"])))
        )
        for ep, start, stop in zip(
            ep_ids,
            meta_episodes["dataset_from_index"],
            meta_episodes["dataset_to_index"],
        ):
            records.append(
                {"episode_index": int(ep), "from": int(start), "to": int(stop)}
            )
    elif isinstance(meta_episodes, list):
        records = [dict(row) for row in meta_episodes]
    elif meta_episodes is not None and hasattr(meta_episodes, "to_list"):
        records = [dict(row) for row in meta_episodes.to_list()]

    dataset_len = len(dataset) if hasattr(dataset, "__len__") else None
    for row in records:
        ep = int(row.get("episode_index", row.get("episode_id", -1)))
        if ep not in wanted:
            continue
        start = row.get("dataset_from_index", row.get("from", row.get("start")))
        stop = row.get("dataset_to_index", row.get("to", row.get("stop")))
        if start is None or stop is None:
            continue
        start_i, stop_i = int(start), int(stop)
        if dataset_len is not None and (start_i < 0 or stop_i > dataset_len):
            # Global metadata for a subset load — skip; do not use these.
            continue
        bounds[ep] = (start_i, stop_i)
    if bounds:
        return bounds

    episode_data_index = getattr(dataset, "episode_data_index", None)
    if isinstance(episode_data_index, dict):
        starts = episode_data_index.get("from", [])
        stops = episode_data_index.get("to", [])
        for position, ep in enumerate(episodes):
            if position < len(starts) and position < len(stops):
                bounds[ep] = (int(starts[position]), int(stops[position]))
    if bounds:
        return bounds

    raise RuntimeError(
        "Could not determine episode frame ranges from this LeRobot dataset. "
        "Inspect dataset.meta.episodes for the installed LeRobot version."
    )


def _task_text(sample: dict[str, Any], dataset: Any) -> str:
    for key in ("task", "language_instruction", "instruction", "task_instruction"):
        value = sample.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()

    task_index = sample.get("task_index")
    tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
    if task_index is not None and tasks is not None:
        try:
            index = int(task_index)
            if isinstance(tasks, dict):
                value = tasks.get(index, tasks.get(str(index)))
            elif isinstance(tasks, (list, tuple)):
                value = tasks[index]
            elif hasattr(tasks, "iloc"):
                value = tasks.iloc[index]
                if hasattr(value, "get"):
                    value = value.get("task", value.iloc[0])
            else:
                value = None
            if isinstance(value, dict):
                value = value.get("task") or value.get("instruction")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    return "complete the requested robot task"


def _open_source(source: dict[str, Any], seed: int):
    dataset_class = _load_lerobot_dataset_class()
    metadata_class = _load_lerobot_metadata_class()
    repo_id = str(source["repo_id"])
    root = source.get("root")
    max_episodes = int(source.get("max_episodes", 2000))

    total_episodes: int | None = None
    if metadata_class is not None:
        try:
            meta = metadata_class(repo_id, root=root)
            total_episodes = int(meta.total_episodes)
        except Exception:
            total_episodes = None

    if total_episodes is not None:
        all_episodes = list(range(total_episodes))
        rng = random.Random(seed + zlib.crc32(repo_id.encode("utf-8")) % 10000)
        rng.shuffle(all_episodes)
        episodes = sorted(all_episodes[:max_episodes])
    else:
        episodes = list(range(max_episodes))

    # SuperPOD often has no system FFmpeg, so torchcodec fails at import time.
    # Prefer pyav (already pulled in by lerobot) for frame decoding.
    kwargs: dict[str, Any] = {
        "root": root,
        "episodes": episodes,
        "download_videos": True,
        "video_backend": "pyav",
    }
    try:
        dataset = dataset_class(repo_id, **kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        try:
            dataset = dataset_class(repo_id, **kwargs)
        except TypeError:
            kwargs.pop("download_videos", None)
            dataset = dataset_class(repo_id, **kwargs)
    return dataset, episodes


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    frames_dir = ensure_dir(cfg["paths"]["frames_dir"])
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    max_rows = int(data_cfg.get("max_rows", 30000))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    threshold = float(data_cfg.get("translation_threshold", 0.003))
    collapse_magnitude = bool(data_cfg.get("collapse_magnitude", False))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for source in data_cfg.get("sources", []):
        if not source.get("enabled", True):
            print(f"Skipping disabled source {source.get('name', source.get('repo_id'))}")
            continue
        name = str(source.get("name") or source["repo_id"]).replace("/", "_")
        print(f"Loading {source['repo_id']} ...")
        dataset, episodes = _open_source(source, seed)
        bounds = _episode_bounds(dataset, episodes)

        shuffled = list(episodes)
        random.Random(seed + len(name)).shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_episodes = set(shuffled[:n_val])

        frames_per_episode = int(source.get("frames_per_episode", 6))
        image_key = str(source.get("image_key", ""))

        for episode in tqdm(episodes, desc=name):
            if len(train_rows) + len(val_rows) >= max_rows:
                break
            if episode not in bounds:
                continue
            start, stop = bounds[episode]
            indices = _evenly_spaced(list(range(start, stop)), frames_per_episode)
            for index in indices:
                if len(train_rows) + len(val_rows) >= max_rows:
                    break
                sample = dataset[index]
                key = image_key
                if not key or key not in sample:
                    candidates = [k for k in sample if k.startswith("observation.images.")]
                    if not candidates:
                        raise KeyError(
                            f"No observation.images.* key in {source['repo_id']}; "
                            f"available keys: {list(sample)[:30]}"
                        )
                    key = candidates[0]

                task = _task_text(sample, dataset)
                action = _to_list(sample["action"])
                image = _image_from_value(sample[key])
                frame_idx = int(sample.get("frame_index", index - start))
                filename = f"{name}_ep{episode:06d}_t{frame_idx:06d}.jpg"
                image_path = Path(frames_dir) / filename
                image.save(image_path, quality=90)

                row = {
                    "source": name,
                    "episode_id": int(episode),
                    "frame_idx": frame_idx,
                    "task": task,
                    "action": action,
                    "image_path": str(image_path),
                    "prompt": build_action_prompt(task),
                    "answer": canonical_action_command(
                        action,
                        translation_threshold=threshold,
                        collapse_magnitude=collapse_magnitude,
                    ),
                }
                if episode in val_episodes:
                    val_rows.append(row)
                else:
                    train_rows.append(row)

        print(f"{name}: cumulative train={len(train_rows)} val={len(val_rows)}")
        if len(train_rows) + len(val_rows) >= max_rows:
            break

    if not train_rows:
        raise SystemExit("No action-command rows were prepared. Check the dataset sources.")

    write_jsonl(processed_dir / "train.jsonl", train_rows)
    write_jsonl(processed_dir / "val.jsonl", val_rows)
    print(f"Wrote {len(train_rows)} train / {len(val_rows)} val rows to {processed_dir}")


if __name__ == "__main__":
    main()
