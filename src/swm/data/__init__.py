"""Data utilities for semantic transitions."""

from .labeling import LabelConfig, build_outcome_string, label_transition, score_from_outcome
from .prompts import ACTION_TEMPLATE, build_prompt, format_action

__all__ = [
    "LabelConfig",
    "build_outcome_string",
    "label_transition",
    "score_from_outcome",
    "ACTION_TEMPLATE",
    "build_prompt",
    "format_action",
]


def get_dataset_cls():
    from .dataset import SemanticTransitionDataset

    return SemanticTransitionDataset
