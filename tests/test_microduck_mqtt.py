import json
from pathlib import Path

import pytest

from swm.microduck.mqtt_control import (
    PRESETS,
    MqttSession,
    MqttVelocityClient,
    VelocityValidationError,
    clamp_and_validate,
    load_session_env,
    resolve_session,
    topic_for,
    velocity_payload,
)
from swm.microduck.planner import candidate_actions


def test_clamp_forward():
    p = clamp_and_validate(0.2, 0.0, 0.0)
    assert p == {"vx": 0.2, "vy": 0.0, "yaw": 0.0}


def test_clamp_caps_vx():
    p = clamp_and_validate(1.0, 0.0, 0.0)
    assert p["vx"] == 0.25


def test_reject_turn_in_place():
    with pytest.raises(VelocityValidationError):
        clamp_and_validate(0.0, 0.0, 0.5)


def test_reject_nonzero_vy():
    with pytest.raises(VelocityValidationError):
        clamp_and_validate(0.2, 0.1, 0.0)


def test_reject_negative_vx():
    with pytest.raises(VelocityValidationError):
        clamp_and_validate(-0.1, 0.0, 0.0)


def test_topic_shape():
    assert topic_for("abc123") == "microduck/demo/abc123/cmd/velocity"
    assert MqttSession(session_id="s1").topic.endswith("/cmd/velocity")


def test_load_session_env(tmp_path: Path):
    env = tmp_path / "session.env"
    env.write_text(
        "SESSION_ID=demo42\nMQTT_HOST=broker.emqx.io\nMQTT_PORT=1883\n",
        encoding="utf-8",
    )
    vars_ = load_session_env(env)
    assert vars_["SESSION_ID"] == "demo42"
    s = resolve_session(session_env=env)
    assert s.session_id == "demo42"
    assert s.host == "broker.emqx.io"
    assert s.port == 1883


def test_dry_run_stream_payloads():
    logged: list[tuple[str, dict]] = []
    session = MqttSession(session_id="test")
    client = MqttVelocityClient(
        session, dry_run=True, on_publish=lambda t, p: logged.append((t, p))
    )
    with client:
        sent = client.stream_velocity(0.2, 0.0, 0.3, duration_s=0.3, rate_hz=10.0)
    assert len(sent) >= 2  # at least one motion + stop
    assert sent[-1] == {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
    assert logged[0][0] == "microduck/demo/test/cmd/velocity"
    assert logged[0][1]["vx"] == 0.2
    # payloads are JSON-serializable
    json.dumps(velocity_payload(0.2, 0.0, 0.3))


def test_preset_stop():
    session = MqttSession(session_id="test")
    client = MqttVelocityClient(session, dry_run=True)
    with client:
        sent = client.send_preset("stop")
    assert sent == [PRESETS["stop"]]


def test_candidates_skip_illegal_pairs():
    cfg = {
        "planner": {
            "vx_values": [0.0, 0.2, -0.1],
            "yaw_values": [-0.5, 0.0, 0.5],
            "max_candidates": 50,
        }
    }
    acts = candidate_actions(cfg)
    # no negative vx
    assert all(a[0] >= 0.0 for a in acts)
    # no turn-in-place
    assert all(not (abs(a[2]) > 1e-6 and a[0] <= 1e-6) for a in acts)
    # vy always 0
    assert all(a[1] == 0.0 for a in acts)
