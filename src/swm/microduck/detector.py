"""CCTV duck detector — YOLO weights from pngwn/microduck-detection-dataset.

Falls back to a bright low-sat blob heuristic tuned for the MuJoCo overhead
view (tiny white MicroDuck on a blue checkerboard). YOLO miss also falls back
so demos stay usable under domain shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass
class Detection:
    cx: float  # normalized [0,1]
    cy: float
    w: float
    h: float
    score: float
    label: str = "microduck"


class DuckDetector:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.25):
        self.weights = Path(weights) if weights else None
        self.conf = conf
        self._model = None
        if self.weights and self.weights.is_file():
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.weights))
            except Exception as exc:  # noqa: BLE001
                print(f"DuckDetector: ultralytics load failed ({exc}); using heuristic.")

    def detect(self, image: Image.Image) -> list[Detection]:
        if self._model is not None:
            dets = self._detect_yolo(image)
            if dets:
                return dets
        return self._detect_heuristic(image)

    def best(self, image: Image.Image) -> Detection | None:
        dets = self.detect(image)
        return max(dets, key=lambda d: d.score) if dets else None

    def _detect_yolo(self, image: Image.Image) -> list[Detection]:
        assert self._model is not None
        # Keep YOLO on CPU so 8GB GPUs (e.g. RTX 3060 Ti) dedicate 100% VRAM to Qwen3-VL.
        res = self._model.predict(
            np.asarray(image.convert("RGB")), conf=self.conf, device="cpu", verbose=False
        )
        out: list[Detection] = []
        if not res:
            return out
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return out
        h, w = int(r0.orig_shape[0]), int(r0.orig_shape[1])
        for box in r0.boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0])
            bw = (xyxy[2] - xyxy[0]) / max(1, w)
            bh = (xyxy[3] - xyxy[1]) / max(1, h)
            cx = ((xyxy[0] + xyxy[2]) / 2) / max(1, w)
            cy = ((xyxy[1] + xyxy[3]) / 2) / max(1, h)
            out.append(Detection(cx=cx, cy=cy, w=bw, h=bh, score=score))
        return out

    def _detect_heuristic(self, image: Image.Image) -> list[Detection]:
        """Bright low-sat MicroDuck body on the blue sim floor.

        Overhead zone frames only have ~20 pure-white pixels on the robot, so
        we cluster high-threshold candidates instead of taking a global bbox.
        """
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)

        for thr, sat_t, minpix in (
            (220, 0.25, 8),
            (200, 0.30, 10),
            (185, 0.35, 12),
            (165, 0.40, 18),
        ):
            mask = (mx > thr) & (sat < sat_t)
            # Drop green circle posts and orange/yellow ball highlights.
            mask &= ~((g > r + 25) & (g > b + 15) & (g > 90))
            mask &= ~((r > 145) & (r > g + 25) & (r > b + 35) & (g < 170))
            ys, xs = np.where(mask)
            if len(xs) < minpix:
                continue
            if len(xs) > 400:
                vals = mx[ys, xs]
                keep = vals >= np.percentile(vals, 70)
                ys, xs = ys[keep], xs[keep]
            best: tuple[int, Detection] | None = None
            for cys, cxs in _cluster_points(ys, xs, gap=4):
                area = int(len(cxs))
                if area < minpix:
                    continue
                bw = float((int(cxs.max()) - int(cxs.min()) + 1) / w)
                bh = float((int(cys.max()) - int(cys.min()) + 1) / h)
                if bw > 0.35 or bh > 0.35 or bw < 0.008 or bh < 0.008:
                    continue
                box_area = max(
                    1,
                    (int(cxs.max()) - int(cxs.min()) + 1)
                    * (int(cys.max()) - int(cys.min()) + 1),
                )
                dens = area / float(box_area)
                if dens < 0.12:
                    continue
                det = Detection(
                    cx=float(cxs.mean() / w),
                    cy=float(cys.mean() / h),
                    w=float(max(bw, 0.04)),
                    h=float(max(bh, 0.04)),
                    score=float(min(0.85, 0.35 + area / 200.0)),
                )
                if best is None or area > best[0]:
                    best = (area, det)
            if best is not None:
                return [best[1]]
        return []


def _cluster_points(
    ys: np.ndarray, xs: np.ndarray, *, gap: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy proximity clusters for sparse bright pixels."""
    if len(xs) == 0:
        return []
    order = np.argsort(ys.astype(np.int64) * 100_000 + xs.astype(np.int64))
    ys = ys[order]
    xs = xs[order]
    used = np.zeros(len(xs), dtype=bool)
    clusters: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(xs)):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        head = 0
        while head < len(members):
            j = members[head]
            head += 1
            yj = int(ys[j])
            xj = int(xs[j])
            for k in range(len(xs)):
                if used[k]:
                    continue
                if abs(int(ys[k]) - yj) <= gap and abs(int(xs[k]) - xj) <= gap:
                    used[k] = True
                    members.append(k)
        idx = np.asarray(members, dtype=np.int64)
        clusters.append((ys[idx], xs[idx]))
    return clusters


def draw_detection(image: Image.Image, det: Detection | None) -> Image.Image:
    from PIL import ImageDraw

    out = image.copy()
    if det is None:
        return out
    draw = ImageDraw.Draw(out)
    w, h = out.size
    x0 = int((det.cx - det.w / 2) * w)
    y0 = int((det.cy - det.h / 2) * h)
    x1 = int((det.cx + det.w / 2) * w)
    y1 = int((det.cy + det.h / 2) * h)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 64, 0), width=3)
    draw.text((x0, max(0, y0 - 12)), f"{det.label} {det.score:.2f}", fill=(255, 64, 0))
    return out


def ball_centroid_heuristic(image: Image.Image) -> tuple[float, float] | None:
    """Normalized (cx, cy) of orange/yellow ball if visible."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    orange = (r > 140) & (g < 120) & (b < 100) & (r > g + 30)
    yellow = (r > 150) & (g > 110) & (b < 120) & (r + g > 2 * b + 40) & (r > b + 40)
    mask = orange | yellow
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    h, w = arr.shape[:2]
    return float(xs.mean() / w), float(ys.mean() / h)
