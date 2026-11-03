from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from swm.data.prompts import build_prompt
from swm.utils.io import load_jsonl


class SemanticTransitionDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, frames_root: str | Path | None = None):
        self.rows = load_jsonl(jsonl_path)
        self.frames_root = Path(frames_root) if frames_root else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_file() and self.frames_root is not None:
            image_path = self.frames_root / image_path.name
        image = Image.open(image_path).convert("RGB")
        # Action-command rows carry their own prompt/answer so the same
        # trainer can learn instruction -> action instead of action -> outcome.
        prompt = row.get("prompt") or build_prompt(row["task"], row["action"])
        answer = row.get("answer") or row["outcome"]
        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "task": row["task"],
            "action": row.get("action", []),
            "episode_id": row.get("episode_id", -1),
        }


def make_collate_fn(processor, max_length: int = 512) -> Callable:
    """Collate for Qwen3-VL style processors (chat template + images)."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []
        for sample in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text", "text": sample["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(sample["image"])

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        # Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask the prompt portion: keep only assistant answer tokens for loss.
        # Heuristic: find "Answer:" marker per row when possible.
        answer_token = processor.tokenizer.encode("Answer:", add_special_tokens=False)
        if answer_token:
            for i in range(labels.size(0)):
                ids = inputs["input_ids"][i].tolist()
                start = _find_subsequence(ids, answer_token)
                if start is not None:
                    # Keep tokens after the Answer: marker
                    end = start + len(answer_token)
                    labels[i, :end] = -100

        inputs["labels"] = labels
        return inputs

    return _collate


def _find_subsequence(seq: list[int], pattern: list[int]) -> int | None:
    n, m = len(seq), len(pattern)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if seq[i : i + m] == pattern:
            return i
    return None
