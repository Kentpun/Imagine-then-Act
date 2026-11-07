"""CCTV duck detector — YOLO weights from pngwn/microduck-detection-dataset.

Falls back to a simple orange/cream color blob heuristic if ultralytics / weights
are unavailable (keeps demos runnable without a trained detector).
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
            return self._detect_yolo(image)
        return self._detect_heuristic(image)

    def best(self, image: Image.Image) -> Detection | None:
        dets = self.detect(image)
        return max(dets, key=lambda d: d.score) if dets else None

    def _detect_yolo(self, image: Image.Image) -> list[Detection]:
        assert self._model is not None
        res = self._model.predict(np.asarray(image.convert("RGB")), conf=self.conf, verbose=False)
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
        """Rough cream/white blob for sim ducks on green floors."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        # bright / low-saturation pixels
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        mask = (mx > 160) & (sat < 0.35)
        ys, xs = np.where(mask)
        if len(xs) < 50:
            return []
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = (x1 - x0) / max(1, w)
        bh = (y1 - y0) / max(1, h)
        if bw < 0.02 or bh < 0.02:
            return []
        return [
            Detection(
                cx=((x0 + x1) / 2) / w,
                cy=((y0 + y1) / 2) / h,
                w=bw,
                h=bh,
                score=0.4,
            )
        ]


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
    draw.text((x0, max(0, y0 - 12)), f"duck {det.score:.2f}", fill=(255, 64, 0))
    return out


def ball_centroid_heuristic(image: Image.Image) -> tuple[float, float] | None:
    """Normalized (cx, cy) of orange ball if visible."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r > 140) & (g < 120) & (b < 100) & (r > g + 30)
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    h, w = arr.shape[:2]
    return float(xs.mean() / w), float(ys.mean() / h)
