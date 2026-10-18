from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image

from swm.data.dataset import SemanticTransitionDataset
from swm.models.semantic_vlm import load_finetuned
from swm.planning.imagination_planner import ImaginationPlanner
from swm.utils.config import load_config


st.set_page_config(page_title="Imagine-then-Act", layout="wide")


@st.cache_resource
def load_runtime(config_path: str, checkpoint: str):
    cfg = load_config(config_path)
    model, processor = load_finetuned(cfg, checkpoint)
    planner = ImaginationPlanner(model, processor, cfg)
    return cfg, planner


def main() -> None:
    st.title("Imagine-then-Act")
    st.caption("Semantic world model — rank actions by imagined outcomes")

    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    checkpoint = st.sidebar.text_input(
        "Checkpoint", "outputs/run001/checkpoint-final"
    )
    if not Path(checkpoint).exists():
        st.warning("Checkpoint not found. Train first with `bash scripts/train.sh`.")
        st.stop()

    cfg, planner = load_runtime(config_path, checkpoint)
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
    task = st.text_input("Task", sample["task"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Observation")
        st.image(sample["image"], use_column_width=True)

    with col2:
        st.subheader("Ranked candidate actions")
        if st.button("Run imagination planner", type="primary"):
            with st.spinner("Scoring candidates..."):
                ranked = planner.rank(sample["image"], task)
            for i, item in enumerate(ranked[:8]):
                a = item.action
                st.markdown(
                    f"**#{i+1} score={item.score:.2f}** — "
                    f"dx={a[0]:.3f}, dy={a[1]:.3f}, dz={a[2]:.3f}, "
                    f"gripper={'close' if a[6] >= 0.5 else 'open'}"
                )
                st.code(item.outcome)

            st.subheader("Action success heatmap (xy)")
            with st.spinner("Building heatmap..."):
                xs, ys, grid = planner.heatmap_xy(sample["image"], task)
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
