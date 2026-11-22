"""Scene- and state-aware counterfactual oracles for Layer B evaluation.

Eliminates the B1 action-only tautology in offline planner evaluation:

1. ``microduck_rollout``: steps the true kinematic scene ``step_scene(s_t, a)``
   for every candidate action ``a`` from ``sample["scene_state"]`` and scores
   true task progress toward the goal/zone.
2. ``state_counterfactual``: grounds LIBERO candidate evaluation in the real
   simulator transition ``(state_t, state_tp1, a_logged, y_true)`` rather than
   action-only rules. Blindly closing the gripper and moving +z in empty space
   no longer receives synthetic ``lifted=yes`` credit unless the scene state
   actually supports grasping/lifting.
3. ``action_only_legacy``: retained only for backward ablation comparison to
   demonstrate the tautology of action-only pseudo-oracles.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from swm.data.labeling import (
    LabelConfig,
    build_outcome_string,
    label_transition,
    score_from_outcome,
)
from swm.microduck.labeling import score_from_outcome as microduck_score_from_outcome
from swm.microduck.semantic_tasks import (
    SceneState,
    TaskKind,
    ball_in_zone,
    facing_goal,
    heading_error,
    label_microduck_transition,
    navigation_target,
    step_scene,
)


def _as7(action: Sequence[float]) -> list[float]:
    a = [float(x) for x in action] + [0.0] * 7
    return a[:7]


def _norm3(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _cosine3(u: Sequence[float], v: Sequence[float]) -> float:
    nu, nv = _norm3(u), _norm3(v)
    if nu <= 1e-8 or nv <= 1e-8:
        return 1.0 if (nu <= 1e-8 and nv <= 1e-8) else 0.0
    return (
        float(u[0]) * float(v[0])
        + float(u[1]) * float(v[1])
        + float(u[2]) * float(v[2])
    ) / (nu * nv)


def parse_outcome_fields(outcome: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (outcome or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out


def state_counterfactual_score(
    action: Sequence[float],
    sample: dict[str, Any],
    cfg: LabelConfig,
    *,
    disabled_fields: Sequence[str] | None = None,
) -> tuple[float, str]:
    """Scene-grounded counterfactual score for LIBERO transitions.

    Uses proprioceptive ``state`` / ``state_next`` when present plus the
    ground-truth transition ``(logged_action, y_true)`` observed in the scene:
      - Exact match to ``logged_action`` yields the empirical simulator outcome
        ``y_true``.
      - Counterfactual candidates are evaluated by whether their 3D motion and
        gripper actuation are physically consistent with the scene's current
        affordance (e.g. ``lifted=yes`` is only reachable if the scene is in a
        graspable state, not during free-space reaching).
    """
    cand = _as7(action)
    logged = _as7(sample.get("action") or [0.0] * 7)
    y_true = str(sample.get("answer") or "")
    true_fields = parse_outcome_fields(y_true)

    # Check if candidate matches the logged demonstrator action
    xyz_diff = math.sqrt(sum((cand[i] - logged[i]) ** 2 for i in range(3)))
    cand_grip = "close" if cand[6] >= 0.65 else ("open" if cand[6] <= 0.35 else "hold")
    log_grip = "close" if logged[6] >= 0.65 else ("open" if logged[6] <= 0.35 else "hold")

    if xyz_diff <= 1e-3 and cand_grip == log_grip and y_true:
        base_score = score_from_outcome(y_true, disabled_fields=disabled_fields)
        # Small tie-break bonus for exact empirical demonstrator transition
        return base_score + 0.25, y_true

    state_t = sample.get("state")
    state_tp1 = sample.get("state_next")
    frame_idx = int(sample.get("frame_idx") or 0)
    episode_len = int(sample.get("episode_len") or max(frame_idx + 20, 100))
    ep_success = bool(
        sample.get("episode_success")
        if sample.get("episode_success") is not None
        else (true_fields.get("success_near") == "yes")
    )

    planar = math.hypot(cand[0], cand[1])
    moved = "yes" if planar > cfg.move_threshold or abs(cand[2]) > cfg.move_threshold else "no"

    # Determine whether the scene currently affords lifting an object:
    # Either the empirical transition lifted, or gripper was already closed at low z.
    grasp_ready = true_fields.get("lifted") == "yes"
    if state_t is not None and len(state_t) >= 3:
        eef_z = float(state_t[2])
        was_closed = len(state_t) > 6 and float(state_t[-1]) >= 0.5
        if was_closed and eef_z < 0.15:
            grasp_ready = True
        if state_tp1 is not None and len(state_tp1) >= 3:
            # Check if scene blocked motion (contact / table limit)
            realized_dz = float(state_tp1[2]) - float(state_t[2])
            if realized_dz > cfg.lift_threshold and log_grip in {"close", "hold"}:
                grasp_ready = True

    lifted = (
        "yes"
        if (grasp_ready and cand[2] > cfg.lift_threshold and cand_grip in {"close", "hold"})
        else "no"
    )

    # Task-progress grounding: compare candidate motion direction & magnitude
    # against the scene's demonstrated task-solving displacement.
    realized_vec: Sequence[float] = logged[:3]
    if (
        state_t is not None
        and state_tp1 is not None
        and len(state_t) >= 3
        and len(state_tp1) >= 3
    ):
        realized_vec = [
            float(state_tp1[0]) - float(state_t[0]),
            float(state_tp1[1]) - float(state_t[1]),
            float(state_tp1[2]) - float(state_t[2]),
        ]

    cos_align = _cosine3(cand[:3], realized_vec)
    grip_match = 1.0 if cand_grip == log_grip else -0.5

    demo_good = true_fields.get("progress") != "down"
    if demo_good:
        if cos_align >= 0.55 and cand_grip == log_grip and _norm3(cand[:3]) >= 0.3 * _norm3(realized_vec):
            progress = "up"
        elif cos_align < -0.25 or (cand_grip != log_grip and log_grip in {"open", "close"}):
            progress = "down"
        else:
            progress = "flat"
    else:
        # Demonstrator stepped downward/regressed; opposite or cautious hold is flat/up
        progress = "down" if cos_align > 0.7 else "flat"

    remaining = episode_len - frame_idx - 1
    success_near = (
        "yes"
        if (
            ep_success
            and remaining <= cfg.success_near_frames
            and progress == "up"
            and cand_grip == log_grip
        )
        else "no"
    )

    outcome = build_outcome_string(
        {
            "moved": moved,
            "lifted": lifted,
            "gripper": cand_grip,
            "progress": progress,
            "success_near": success_near,
        }
    )
    discrete = score_from_outcome(outcome, disabled_fields=disabled_fields)
    # Continuous shaping term breaks grid ties using actual scene trajectory alignment
    shaping = 0.5 * cos_align + 0.15 * grip_match - 2.0 * min(0.25, xyz_diff)
    return discrete + shaping, outcome


def _scene_from_dict(d: dict[str, Any]) -> SceneState:
    return SceneState(
        duck_x=float(d.get("duck_x", 0.0)),
        duck_y=float(d.get("duck_y", 0.0)),
        duck_yaw=float(d.get("duck_yaw", 0.0)),
        ball_x=float(d.get("ball_x", 0.0)),
        ball_y=float(d.get("ball_y", 0.0)),
        zone_cx=float(d.get("zone_cx", 0.0)),
        zone_cy=float(d.get("zone_cy", 0.0)),
        zone_r=float(d.get("zone_r", 0.35)),
        goal_x=float(d.get("goal_x", 1.2)),
        goal_y=float(d.get("goal_y", 0.0)),
        zone_b_x=float(d.get("zone_b_x", 1.0)),
        zone_b_y=float(d.get("zone_b_y", 0.5)),
    )


def _infer_microduck_task(sample: dict[str, Any]) -> TaskKind:
    tk = sample.get("task_kind")
    if tk in ("walk_ab", "kick_goal", "keep_zone", "conditional"):
        return tk  # type: ignore[return-value]
    text = str(sample.get("task", "")).lower()
    if "zone b" in text or "walk from" in text:
        return "walk_ab"
    if "kick" in text or "goal" in text:
        return "kick_goal"
    if "if " in text or "else hold" in text:
        return "conditional"
    return "keep_zone"


def _microduck_potential(task: TaskKind, s: SceneState) -> float:
    """Higher potential = closer to task completion in continuous scene coords."""
    if task == "walk_ab":
        dist = math.hypot(s.duck_x - s.zone_b_x, s.duck_y - s.zone_b_y)
        align = 0.15 if facing_goal(s, s.zone_b_x, s.zone_b_y) else 0.0
        bonus = 2.0 if dist < 0.3 else 0.0
        return -2.5 * dist + align + bonus

    if task == "kick_goal":
        d_goal = math.hypot(s.ball_x - s.goal_x, s.ball_y - s.goal_y)
        d_ball = math.hypot(s.duck_x - s.ball_x, s.duck_y - s.ball_y)
        align = 0.2 if facing_goal(s, s.goal_x, s.goal_y, tol=0.8) else 0.0
        bonus = 3.0 if d_goal < 0.35 else 0.0
        return -3.0 * d_goal - 0.8 * d_ball + align + bonus

    # keep_zone / conditional
    d_zone = max(0.0, math.hypot(s.ball_x - s.zone_cx, s.ball_y - s.zone_cy) - s.zone_r)
    d_ball = math.hypot(s.duck_x - s.ball_x, s.duck_y - s.ball_y)
    inside = ball_in_zone(s)
    if inside:
        return 2.5
    return -3.0 * d_zone - 0.6 * d_ball


def microduck_rollout_score(
    action: Sequence[float],
    sample: dict[str, Any],
) -> tuple[float, str]:
    """True 1-step kinematic simulator rollout oracle for MicroDuck."""
    scene_dict = sample.get("scene_state")
    if not scene_dict:
        # Fallback to state_counterfactual if scene_state wasn't stored
        return state_counterfactual_score(action, sample, LabelConfig())

    s0 = _scene_from_dict(scene_dict)
    task = _infer_microduck_task(sample)

    a = [float(x) for x in action] + [0.0] * 7
    # Support both [vx, 0, yaw] (3D planner) and [vx, yaw, 0,0,0,0, grip] (7D dataset)
    if len(action) == 3:
        vx = a[0]
        yaw = a[2]
        is_kick = abs(yaw - 9.0) < 1e-6
    else:
        vx = a[0]
        yaw = a[1] if abs(a[2]) < 1e-6 else a[2]
        is_kick = a[6] >= 0.5 or abs(yaw - 9.0) < 1e-6

    skill = "kick" if is_kick else ("walk" if abs(vx) > 0.05 else "hold")
    s1 = step_scene(
        s0,
        0.0 if is_kick else max(0.0, vx),
        0.0 if is_kick else yaw,
        skill=skill,
    )

    pot0 = _microduck_potential(task, s0)
    pot1 = _microduck_potential(task, s1)
    delta = pot1 - pot0

    outcome = label_microduck_transition(
        task,
        s0,
        s1,
        0.0 if is_kick else max(0.0, vx),
        0.0 if is_kick else yaw,
        skill=skill,
    )
    sem_score = microduck_score_from_outcome(outcome, task=str(sample.get("task", "")))
    tx, ty = navigation_target(task, s0)
    ang_gain = abs(heading_error(s0, tx, ty)) - abs(heading_error(s1, tx, ty))
    total_score = sem_score + 1.5 * delta + 0.8 * ang_gain + 0.02 * max(0.0, vx)
    return total_score, outcome


def action_only_legacy_score(
    action: Sequence[float],
    cfg: LabelConfig,
    *,
    disabled_fields: Sequence[str] | None = None,
) -> tuple[float, str]:
    """Legacy action-only pseudo-oracle (tautological with B1_heuristic)."""
    labels = label_transition(
        _as7(action),
        None,
        None,
        frame_idx=0,
        episode_len=100,
        episode_success=False,
        cfg=cfg,
    )
    outcome = build_outcome_string(labels)
    return score_from_outcome(outcome, disabled_fields=disabled_fields), outcome
