"""Load lerobot/libero v3.0 locally: parquet states/actions + MP4 camera frames.

HuggingFace `load_dataset` only returns parquet columns (no images). Camera
streams live under ``videos/{key}/chunk-XXX/file-YYY.mp4`` and must be decoded
with episode ``from_timestamp`` + per-frame ``timestamp``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_VIDEO_KEY = "observation.images.image"


def _read_parquet_table(path: Path):
    return pq.read_table(path).to_pydict()


def _concat_parquet_dicts(paths: list[Path]) -> dict[str, list]:
    if not paths:
        return {}
    cols: dict[str, list] = {}
    for path in paths:
        part = _read_parquet_table(path)
        if not cols:
            cols = {k: list(v) for k, v in part.items()}
        else:
            for k, v in part.items():
                cols[k].extend(list(v))
    return cols


def _as_float(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _as_int(x: Any) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


class LiberoV3FrameDataset:
    """Indexable frame dataset with decoded RGB under ``observation.images.image``."""

    def __init__(
        self,
        root: str | Path,
        *,
        video_key: str = DEFAULT_VIDEO_KEY,
    ):
        self.root = Path(root)
        self.video_key = video_key
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing {info_path}")
        videos_dir = self.root / "videos" / video_key
        if not videos_dir.is_dir():
            raise FileNotFoundError(
                f"Missing camera videos at {videos_dir}. "
                "Re-download with scripts/download_data.py so the videos/ tree is present."
            )

        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.video_path_tmpl = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )

        ep_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"No episode parquet under {self.root / 'meta' / 'episodes'}")
        ep_cols = _concat_parquet_dicts(ep_files)
        self.episodes: dict[int, dict[str, Any]] = {}
        n_ep = len(ep_cols["episode_index"])
        for i in range(n_ep):
            ep_id = _as_int(ep_cols["episode_index"][i])
            self.episodes[ep_id] = {k: ep_cols[k][i] for k in ep_cols}

        task_path = self.root / "meta" / "tasks.parquet"
        self.tasks: dict[int, str] = {}
        if task_path.is_file():
            tdf = pq.read_table(task_path).to_pandas()
            # LeRobot stores task string as the DataFrame index.
            for task_str, row in tdf.iterrows():
                self.tasks[_as_int(row["task_index"])] = str(task_str)

        data_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No data parquet under {self.root / 'data'}")
        self._cols = _concat_parquet_dicts(data_files)
        self._n = len(self._cols["index"])

        # Cache last opened video for sequential episode reads.
        self._av_path: Path | None = None
        self._av_container = None

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        if self._av_container is not None:
            self._av_container.close()
            self._av_container = None
            self._av_path = None

    def _video_file(self, ep_meta: dict[str, Any]) -> Path:
        prefix = f"videos/{self.video_key}/"
        chunk = _as_int(ep_meta[f"{prefix}chunk_index"])
        file_index = _as_int(ep_meta[f"{prefix}file_index"])
        rel = self.video_path_tmpl.format(
            video_key=self.video_key,
            chunk_index=chunk,
            file_index=file_index,
        )
        return self.root / rel

    def _decode_frame(self, path: Path, t_sec: float) -> Image.Image:
        import av

        if self._av_path != path:
            self.close()
            self._av_container = av.open(str(path))
            self._av_path = path

        container = self._av_container
        assert container is not None
        stream = container.streams.video[0]
        # Seek in stream time_base units; decode forward to target time.
        tb = stream.time_base
        target_pts = int(t_sec / float(tb))
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)

        best = None
        best_t = -1.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ft = float(frame.pts * tb)
            best = frame
            best_t = ft
            if ft >= t_sec - 1e-3:
                break
        if best is None:
            raise RuntimeError(f"No frame decoded from {path} at t={t_sec:.4f}s")
        arr = best.to_ndarray(format="rgb24")
        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        ep = _as_int(self._cols["episode_index"][idx])
        if ep not in self.episodes:
            raise KeyError(f"episode_index={ep} missing from meta/episodes")
        ep_meta = self.episodes[ep]
        prefix = f"videos/{self.video_key}/"
        from_ts = _as_float(ep_meta[f"{prefix}from_timestamp"])
        ts = _as_float(self._cols["timestamp"][idx])
        video_path = self._video_file(ep_meta)
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        image = self._decode_frame(video_path, from_ts + ts)
        task_index = _as_int(self._cols["task_index"][idx])
        task = self.tasks.get(task_index, "complete the manipulation task")

        state = self._cols["observation.state"][idx]
        action = self._cols["action"][idx]
        return {
            "observation.images.image": image,
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "timestamp": ts,
            "frame_index": _as_int(self._cols["frame_index"][idx]),
            "episode_index": ep,
            "index": _as_int(self._cols["index"][idx]),
            "task_index": task_index,
            "task": task,
        }
