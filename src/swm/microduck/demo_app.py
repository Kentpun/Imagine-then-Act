from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from swm.microduck.mqtt_control import (
    PRESETS,
    MqttVelocityClient,
    VelocityValidationError,
    resolve_session,
)
from swm.microduck.planner import MicroDuckImaginationPlanner, RankedAction
from swm.utils.config import load_config
from swm.utils.io import load_jsonl


BROWSER_SIM = "https://huggingface.co/spaces/pollen-robotics/microduck-simulator"
DOCS = "https://microduck.homes/docs"
DEMO_REPO = "https://github.com/hjianbo/microduck_demo_part1"


st.set_page_config(page_title="Imagine-then-Act · MicroDuck", layout="wide")


@st.cache_resource
def load_vlm_planner(config_path: str, checkpoint: str):
    from swm.models.semantic_vlm import load_finetuned

    cfg = load_config(config_path)
    model, processor = load_finetuned(cfg, checkpoint)
    planner = MicroDuckImaginationPlanner(
        cfg, model=model, processor=processor, mock=False
    )
    return cfg, planner


def load_mock_planner(config_path: str):
    cfg = load_config(config_path)
    return cfg, MicroDuckImaginationPlanner(cfg, mock=True)


def _mqtt_sidebar(cfg: dict) -> dict:
    mqtt_cfg = cfg.get("mqtt", {})
    st.sidebar.markdown("---")
    st.sidebar.subheader("MQTT control")
    enabled = st.sidebar.toggle(
        "Enable MQTT publish",
        value=bool(mqtt_cfg.get("enabled", False)),
    )
    dry_run = st.sidebar.toggle(
        "Dry-run (log only)",
        value=bool(mqtt_cfg.get("dry_run", True)),
        disabled=not enabled,
    )
    session_env = st.sidebar.text_input(
        "session.env path",
        value=str(mqtt_cfg.get("session_env") or ".demo/session.env"),
    )
    host = st.sidebar.text_input(
        "Broker host", value=str(mqtt_cfg.get("host") or "broker.emqx.io")
    )
    port = int(
        st.sidebar.number_input(
            "Broker port",
            min_value=1,
            max_value=65535,
            value=int(mqtt_cfg.get("port") or 1883),
        )
    )
    session_id = st.sidebar.text_input(
        "Session ID (blank → session.env / env)",
        value=str(mqtt_cfg.get("session_id") or ""),
    )
    duration_s = float(
        st.sidebar.number_input(
            "Stream duration (s)",
            min_value=0.5,
            max_value=30.0,
            value=float(mqtt_cfg.get("duration_s") or 3.0),
            step=0.5,
        )
    )
    rate_hz = float(
        st.sidebar.number_input(
            "Publish rate (Hz)",
            min_value=1.0,
            max_value=50.0,
            value=float(mqtt_cfg.get("rate_hz") or 10.0),
            step=1.0,
        )
    )
    return {
        "enabled": enabled,
        "dry_run": dry_run if enabled else True,
        "session_env": session_env,
        "host": host,
        "port": port,
        "session_id": session_id or None,
        "duration_s": duration_s,
        "rate_hz": rate_hz,
    }


def _make_client(mqtt_ui: dict) -> MqttVelocityClient:
    session = resolve_session(
        session_env=mqtt_ui["session_env"],
        host=mqtt_ui["host"],
        port=mqtt_ui["port"],
        session_id=mqtt_ui["session_id"],
    )
    return MqttVelocityClient(
        session,
        dry_run=bool(mqtt_ui["dry_run"]) or not bool(mqtt_ui["enabled"]),
    )


def _execute_velocity(
    mqtt_ui: dict, vx: float, vy: float, yaw: float, *, label: str
) -> None:
    if not mqtt_ui["enabled"] and not mqtt_ui["dry_run"]:
        st.warning("Enable MQTT or dry-run to publish.")
        return
    client = _make_client(mqtt_ui)
    try:
        with client:
            sent = client.stream_velocity(
                vx,
                vy,
                yaw,
                duration_s=mqtt_ui["duration_s"],
                rate_hz=mqtt_ui["rate_hz"],
            )
        mode = "dry-run" if client.dry_run else "MQTT"
        st.success(
            f"{label}: {mode} → `{client.topic}`  "
            f"last={sent[-1] if sent else {}}  (n={len(sent)})"
        )
        st.code(
            f"topic={client.topic}\n"
            f"broker={client.session.host}:{client.session.port}\n"
            f"payload_example={sent[0] if sent else {}}"
        )
    except VelocityValidationError as e:
        st.error(f"Illegal velocity: {e}")
    except Exception as e:  # noqa: BLE001 — surface MQTT/runtime errors in UI
        st.error(f"MQTT publish failed: {e}")


def main() -> None:
    st.title("Imagine-then-Act · Part B (MicroDuck)")
    st.caption(
        "Imagine velocities → rank → MQTT-publish intents to MuJoCo "
        "(microduck_demo_part1). Walking is still the ONNX policy."
    )

    st.info(
        f"**B0 — Browser sim:** [{BROWSER_SIM}]({BROWSER_SIM})  \n"
        f"**B1c — MQTT MuJoCo control:** [{DEMO_REPO}]({DEMO_REPO})  \n"
        f"Docs: [{DOCS}]({DOCS})"
    )

    default_mock = os.environ.get("SWM_MICRODUCK_MOCK", "1") != "0"
    default_ckpt = os.environ.get(
        "SWM_MICRODUCK_CHECKPOINT", "outputs/run001/checkpoint-final"
    )

    config_path = st.sidebar.text_input("Config", "configs/microduck_demo.yaml")
    mock = st.sidebar.toggle("Mock world model (no GPU)", value=default_mock)
    checkpoint = st.sidebar.text_input("Part A checkpoint (optional)", default_ckpt)

    if mock:
        cfg, planner = load_mock_planner(config_path)
        st.sidebar.success("Mock mode — deterministic heuristic outcomes")
    else:
        if not Path(checkpoint).exists():
            st.warning("Checkpoint missing. Enable mock mode or train Part A first.")
            st.stop()
        cfg, planner = load_vlm_planner(config_path, checkpoint)
        st.sidebar.warning(
            "Using Part A weights zero-shot on MicroDuck prompts — expect weak semantics."
        )

    mqtt_ui = _mqtt_sidebar(cfg)

    smoke_dir = Path(cfg["paths"]["smoke_dir"])
    jsonl = smoke_dir / "samples.jsonl"
    if not jsonl.is_file():
        st.error(
            "Smoke data missing. Run: `python scripts/prepare_microduck_smoke.py`"
        )
        st.stop()

    rows = load_jsonl(jsonl)
    idx = st.sidebar.number_input(
        "Sample index", min_value=0, max_value=max(0, len(rows) - 1), value=0
    )
    row = rows[int(idx)]
    image = Image.open(row["image_path"]).convert("RGB")
    task = st.text_input("Task", row.get("task", "walk forward without falling"))

    st.subheader("Teleop presets")
    t1, t2, t3, t4 = st.columns(4)
    if t1.button("Forward"):
        p = PRESETS["forward"]
        _execute_velocity(mqtt_ui, p["vx"], p["vy"], p["yaw"], label="Forward")
    if t2.button("Forward-left"):
        p = PRESETS["forward-left"]
        _execute_velocity(mqtt_ui, p["vx"], p["vy"], p["yaw"], label="Forward-left")
    if t3.button("Forward-right"):
        p = PRESETS["forward-right"]
        _execute_velocity(mqtt_ui, p["vx"], p["vy"], p["yaw"], label="Forward-right")
    if t4.button("Stop"):
        client = _make_client(mqtt_ui)
        try:
            with client:
                payload = client.publish_velocity(0.0, 0.0, 0.0)
            mode = "dry-run" if client.dry_run else "MQTT"
            st.success(f"Stop: {mode} → `{client.topic}`  {payload}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Stop failed: {e}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Observation (smoke / capture)")
        st.image(image, use_container_width=True)
        st.code(row.get("outcome", ""), language="text")

    with col2:
        st.subheader("Ranked velocity commands")
        if st.button("Run imagination planner", type="primary"):
            with st.spinner("Scoring candidates..."):
                ranked = planner.rank(image, task)
            st.session_state["microduck_ranked"] = [
                {"action": r.action, "outcome": r.outcome, "score": r.score}
                for r in ranked
            ]

        ranked_raw = st.session_state.get("microduck_ranked")
        if ranked_raw:
            ranked = [
                RankedAction(
                    action=list(r["action"]),
                    outcome=str(r["outcome"]),
                    score=float(r["score"]),
                )
                for r in ranked_raw
            ]
            for i, item in enumerate(ranked[:8]):
                a = item.action
                st.markdown(
                    f"**#{i+1} score={item.score:.2f}** — "
                    f"vx={a[0]:.2f}, vy={a[1]:.2f}, yaw={a[2]:.2f}"
                )
                st.code(item.outcome)

            best = ranked[0]
            if st.button("Execute best on MicroDuck", type="primary"):
                a = best.action
                _execute_velocity(
                    mqtt_ui,
                    float(a[0]),
                    float(a[1]) if len(a) > 1 else 0.0,
                    float(a[2]) if len(a) > 2 else 0.0,
                    label=f"Best (score={best.score:.2f})",
                )

            st.subheader("Heatmap (vx × yaw_rate)")
            with st.spinner("Building heatmap..."):
                xs, ys, grid = planner.heatmap_vx_yaw(image, task)
            fig = go.Figure(
                data=go.Heatmap(
                    z=grid, x=[f"{x:.2f}" for x in xs], y=[f"{y:.2f}" for y in ys]
                )
            )
            fig.update_layout(
                xaxis_title="vx",
                yaxis_title="yaw_rate",
                height=420,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Part B scope"):
        st.markdown(
            f"""
- **Imagine:** rank legal `(vx, yaw)` with mock or VLM world model.
- **Act:** publish JSON velocity intents over MQTT to
  [`microduck_demo_part1`]({DEMO_REPO}) (MuJoCo + ONNX gait).
- **Not claimed:** replacing the walking policy with Qwen; real hardware / Device Agent.
- Spec: `docs/MICRODUCK_DEMO.md`
            """
        )


if __name__ == "__main__":
    main()
