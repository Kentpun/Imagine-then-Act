"""Streamlit demo: Observe → Imagine (K counterfactuals) → Act."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from swm.data.dataset import SemanticTransitionDataset
from swm.models.semantic_vlm import load_finetuned
from swm.planning.imagine_then_act import ImagineThenAct
from swm.utils.config import load_config


st.set_page_config(page_title="Imagine-then-Act", layout="wide")


@st.cache_resource
def load_runtime(config_path: str, checkpoint: str):
    cfg = load_config(config_path)
    model, processor = load_finetuned(cfg, checkpoint)
    controller = ImagineThenAct(model, processor, cfg)
    return cfg, controller


def main() -> None:
    st.title("Imagine-then-Act")
    st.markdown(
        """
**Imagine** = ask the LoRA world model *what would happen* under many candidate
actions on this same image (counterfactual outcomes).  
**Act** = execute / display the action with the best imagined score.

This is model-based planning, not ReAct-style chain-of-thought.
"""
    )

    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    checkpoint = st.sidebar.text_input(
        "Checkpoint", "outputs/superpod/checkpoint-final"
    )
    if not Path(checkpoint).exists():
        st.warning(
            "Checkpoint not found. Train first, or set path to "
            "`outputs/superpod/checkpoint-final`."
        )
        st.stop()

    cfg, controller = load_runtime(config_path, checkpoint)
    ds = SemanticTransitionDataset(
        Path(cfg["paths"]["processed_dir"]) / "val.jsonl",
        Path(cfg["paths"]["frames_dir"]),
    )
    if len(ds) == 0:
        st.error("No val samples. Run prepare_dataset.py first.")
        st.stop()

    idx = st.sidebar.number_input(
        "Sample index",
        min_value=0,
        max_value=max(0, len(ds) - 1),
        value=int(cfg["demo"].get("sample_index", 0)),
    )
    sample = ds[int(idx)]
    task = st.text_input("Task (goal g)", sample["task"])
    top_k = st.sidebar.slider("Show top-K imaginations", 3, 16, 8)

    st.header("1. Observe")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(sample["image"], use_container_width=True, caption="o_t")
    with col2:
        st.write(f"**Goal g:** {task}")
        st.write(
            f"Candidates to imagine: **{len(controller.planner.candidates)}** "
            "(xy grid × dz × gripper)"
        )
        logged = sample.get("action") or []
        if logged:
            st.caption(
                f"Dataset logged action (not used for imagination): "
                f"dx={float(logged[0]):.3f}, dy={float(logged[1]):.3f}, "
                f"dz={float(logged[2]):.3f}"
            )

    st.header("2. Imagine  →  3. Act")
    if st.button("Run Imagine-then-Act", type="primary"):
        with st.spinner("Imagining outcomes for each candidate action..."):
            result = controller.imagine_then_act(sample["image"], task, horizon=1)

        st.subheader("Imagined outcomes (counterfactual)")
        for i, step in enumerate(result.imaginations[:top_k]):
            a = step.action
            mark = " ← **ACT**" if i == 0 else ""
            st.markdown(
                f"**#{i+1} score={step.score:.2f}**{mark} — "
                f"dx={a[0]:.3f}, dy={a[1]:.3f}, dz={a[2]:.3f}, "
                f"gripper={'close' if a[6] >= 0.5 else 'open'}"
            )
            st.code(step.imagined_outcome)

        chosen = result.chosen
        st.success(
            f"**Act:** dx={chosen.action[0]:.3f}, dy={chosen.action[1]:.3f}, "
            f"dz={chosen.action[2]:.3f}, "
            f"gripper={'close' if chosen.action[6] >= 0.5 else 'open'} "
            f"(score={chosen.score:.2f})"
        )

        st.subheader("Imagination heatmap (score over dx, dy)")
        with st.spinner("Building heatmap of imagined scores..."):
            xs, ys, grid = controller.planner.heatmap_xy(sample["image"], task)
        fig = go.Figure(
            data=go.Heatmap(z=grid, x=[str(x) for x in xs], y=[str(y) for y in ys])
        )
        fig.update_layout(
            xaxis_title="dx",
            yaxis_title="dy",
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
