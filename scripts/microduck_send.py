#!/usr/bin/env python3
"""Publish MicroDuck velocity intents over MQTT (EMQX Part 1 protocol).

Requires the MuJoCo side from hjianbo/microduck_demo_part1 running with the
same SESSION_ID / session.env. This script only publishes; it does not open sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swm.microduck.mqtt_control import (  # noqa: E402
    PRESETS,
    MqttVelocityClient,
    VelocityValidationError,
    resolve_session,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "preset",
        nargs="?",
        choices=sorted(PRESETS.keys()),
        help="Named motion preset (optional if --vx/--yaw given)",
    )
    p.add_argument("--vx", type=float, default=None)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=None)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument(
        "--session-env",
        type=str,
        default=".demo/session.env",
        help="Path to microduck_demo_part1 session.env",
    )
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--session-id", type=str, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print payloads without connecting",
    )
    args = p.parse_args()

    if args.preset is None and args.vx is None:
        p.error("Provide a preset (e.g. forward) or --vx")

    session = resolve_session(
        session_env=args.session_env,
        host=args.host,
        port=args.port,
        session_id=args.session_id,
    )
    client = MqttVelocityClient(session, dry_run=args.dry_run)

    try:
        with client:
            if args.preset is not None:
                sent = client.send_preset(
                    args.preset,
                    duration_s=args.duration,
                    rate_hz=args.rate,
                    vx=args.vx,
                    yaw=args.yaw,
                )
            else:
                sent = client.stream_velocity(
                    float(args.vx),
                    float(args.vy),
                    float(args.yaw or 0.0),
                    duration_s=args.duration,
                    rate_hz=args.rate,
                )
    except VelocityValidationError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"error: MQTT publish failed: {e}", file=sys.stderr)
        sys.exit(1)

    mode = "dry-run" if args.dry_run else "mqtt"
    print(f"[{mode}] topic={client.topic} broker={session.host}:{session.port}")
    print(f"sent {len(sent)} messages; last={sent[-1] if sent else {}}")


if __name__ == "__main__":
    main()
