"""MQTT velocity publisher for MicroDuck (EMQX Part 1 protocol).

Publishes movement *intents* compatible with hjianbo/microduck_demo_part1:
  Topic: microduck/demo/{sessionId}/cmd/velocity
  Payload: {"vx": float, "vy": float, "yaw": float}

The ONNX walking policy + MuJoCo sim run outside this repo.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# EMQX demo-validated ranges (BEST_alpha_walking.onnx)
VX_MIN = 0.0
VX_MAX = 0.25
VY_FIXED = 0.0
YAW_MIN = -0.8
YAW_MAX = 0.8
YAW_EPS = 1e-6

DEFAULT_BROKER = "broker.emqx.io"
DEFAULT_PORT = 1883
DEFAULT_RATE_HZ = 10.0

PRESETS: dict[str, dict[str, float]] = {
    "forward": {"vx": 0.25, "vy": 0.0, "yaw": 0.0},
    "forward-left": {"vx": 0.2, "vy": 0.0, "yaw": 0.5},
    "forward-right": {"vx": 0.2, "vy": 0.0, "yaw": -0.5},
    "stop": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
}


class VelocityValidationError(ValueError):
    """Raised when a velocity command violates demo policy limits."""


@dataclass
class MqttSession:
    host: str = DEFAULT_BROKER
    port: int = DEFAULT_PORT
    session_id: str = "local"
    username: str | None = None
    password: str | None = None

    @property
    def topic(self) -> str:
        return f"microduck/demo/{self.session_id}/cmd/velocity"


def load_session_env(path: str | Path | None = None) -> dict[str, str]:
    """Parse KEY=VALUE lines from a session.env file (no export prefix required)."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def resolve_session(
    *,
    session_env: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
    session_id: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> MqttSession:
    """Merge session.env, process env, and explicit kwargs (kwargs win)."""
    file_vars = load_session_env(session_env)
    env = {**file_vars, **dict(os.environ)}

    sid = (
        session_id
        or env.get("SESSION_ID")
        or env.get("MQTT_SESSION_ID")
        or "local"
    )
    h = host or env.get("MQTT_HOST") or env.get("MQTT_BROKER") or DEFAULT_BROKER
    p_raw = port if port is not None else env.get("MQTT_PORT")
    p = int(p_raw) if p_raw is not None else DEFAULT_PORT
    user = username if username is not None else env.get("MQTT_USERNAME")
    pwd = password if password is not None else env.get("MQTT_PASSWORD")
    return MqttSession(host=h, port=p, session_id=sid, username=user, password=pwd)


def clamp_and_validate(
    vx: float, vy: float = 0.0, yaw: float = 0.0, *, strict: bool = True
) -> dict[str, float]:
    """Clamp and validate a velocity intent for the EMQX MicroDuck demo policy.

    Rules:
    - vx in [0, 0.25]
    - vy forced to 0
    - yaw in [-0.8, 0.8]; non-zero yaw requires vx > 0
    """
    vx_c = float(max(VX_MIN, min(VX_MAX, vx)))
    yaw_c = float(max(YAW_MIN, min(YAW_MAX, yaw)))
    vy_c = VY_FIXED

    if abs(vy) > YAW_EPS and strict:
        raise VelocityValidationError(
            f"vy must be 0 for the current walking policy (got {vy})"
        )
    if abs(yaw_c) > YAW_EPS and vx_c <= YAW_EPS:
        if strict:
            raise VelocityValidationError(
                "yaw != 0 requires positive vx (demo policy cannot turn in place)"
            )
        yaw_c = 0.0
    if vx < VX_MIN - YAW_EPS and strict:
        raise VelocityValidationError(f"vx must be >= 0 (got {vx})")

    return {"vx": vx_c, "vy": vy_c, "yaw": yaw_c}


def velocity_payload(vx: float, vy: float = 0.0, yaw: float = 0.0) -> dict[str, float]:
    return clamp_and_validate(vx, vy, yaw, strict=True)


def topic_for(session_id: str) -> str:
    return f"microduck/demo/{session_id}/cmd/velocity"


class MqttVelocityClient:
    """Publish MicroDuck velocity intents over MQTT (or dry-run)."""

    def __init__(
        self,
        session: MqttSession,
        *,
        dry_run: bool = False,
        on_publish: Callable[[str, dict[str, float]], None] | None = None,
    ):
        self.session = session
        self.dry_run = dry_run
        self.on_publish = on_publish
        self._client: Any = None

    @property
    def topic(self) -> str:
        return self.session.topic

    def connect(self) -> None:
        if self.dry_run:
            return
        import paho.mqtt.client as mqtt

        client_id = f"swm-microduck-{self.session.session_id}"
        # paho-mqtt 2.x requires CallbackAPIVersion; 1.x uses positional client_id.
        try:
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
            )
        except (AttributeError, TypeError):
            self._client = mqtt.Client(client_id=client_id)
        if self.session.username:
            self._client.username_pw_set(
                self.session.username, self.session.password or ""
            )
        self._client.connect(self.session.host, self.session.port, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def __enter__(self) -> MqttVelocityClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()

    def publish_velocity(
        self, vx: float, vy: float = 0.0, yaw: float = 0.0
    ) -> dict[str, float]:
        payload = velocity_payload(vx, vy, yaw)
        if self.on_publish is not None:
            self.on_publish(self.topic, payload)
        if self.dry_run:
            return payload
        if self._client is None:
            raise RuntimeError("MQTT client not connected; call connect() first")
        body = json.dumps(payload)
        info = self._client.publish(self.topic, body, qos=0, retain=False)
        info.wait_for_publish(timeout=5.0)
        return payload

    def stream_velocity(
        self,
        vx: float,
        vy: float = 0.0,
        yaw: float = 0.0,
        *,
        duration_s: float = 3.0,
        rate_hz: float = DEFAULT_RATE_HZ,
        stop_after: bool = True,
    ) -> list[dict[str, float]]:
        """Publish the same intent at ``rate_hz`` for ``duration_s``, then stop."""
        period = 1.0 / max(rate_hz, 1e-3)
        n = max(1, int(round(duration_s * rate_hz)))
        sent: list[dict[str, float]] = []
        t0 = time.monotonic()
        for i in range(n):
            sent.append(self.publish_velocity(vx, vy, yaw))
            target = t0 + (i + 1) * period
            sleep_for = target - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        if stop_after:
            sent.append(self.publish_velocity(0.0, 0.0, 0.0))
        return sent

    def send_preset(
        self,
        name: str,
        *,
        duration_s: float = 3.0,
        rate_hz: float = DEFAULT_RATE_HZ,
        vx: float | None = None,
        yaw: float | None = None,
    ) -> list[dict[str, float]]:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
        base = dict(PRESETS[name])
        if vx is not None:
            base["vx"] = float(vx)
        if yaw is not None:
            base["yaw"] = float(yaw)
        if name == "stop":
            return [self.publish_velocity(**base)]
        return self.stream_velocity(
            base["vx"], base["vy"], base["yaw"], duration_s=duration_s, rate_hz=rate_hz
        )
