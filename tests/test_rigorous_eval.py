"""Tests for upgrades 1–3: closed-loop eval, direct-policy baseline, and non-tautological oracles."""

from __future__ import annotations

import random
from pathlib import Path

from swm.data.labeling import LabelConfig
from swm.eval.libero_closed_loop import SyntheticTabletopEnv, _to_robosuite_osc, run_synthetic_tabletop
from swm.eval.oracles import (
    action_only_legacy_score,
    microduck_rollout_score,
    state_counterfactual_score,
)
from swm.eval.planner_eval import pick_top, score_heuristic, score_oracle_candidates
from swm.models.direct_policy import (
    action_distance,
    format_libero_policy_target,
    format_microduck_policy_target,
    parse_libero_policy_action,
    parse_microduck_policy_action,
)


def test_state_counterfactual_breaks_b1_tautology() -> None:
    """In free-space reaching (gripper open, moving +x), blind +z/close heuristic is NOT optimal."""
    cfg = LabelConfig()
    sample = {
        "task": "reach and open gripper above the bowl",
        "action": [0.05, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],  # reaching down/forward with open gripper
        "answer": "moved=yes; lifted=no; gripper=open; progress=up; success_near=no",
        "state": [0.10, 0.00, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state_next": [0.15, 0.00, 0.23, 0.0, 0.0, 0.0, 0.0, 0.0],
        "frame_idx": 5,
        "episode_len": 60,
        "episode_success": True,
    }
    blind_lift = [0.05, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]
    true_reach = [0.05, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0]
    candidates = [blind_lift, true_reach, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    # Legacy action-only oracle tautologically picks blind_lift == B1_heuristic
    legacy_b1 = pick_top(score_heuristic(candidates, cfg))
    leg_oracle, _ = score_oracle_candidates(candidates, sample, cfg, oracle_mode="action_only_legacy")
    assert legacy_b1.action == pick_top(leg_oracle).action == blind_lift

    # New state_counterfactual oracle rejects fake mid-air lift and prefers true_reach
    sc_oracle, used = score_oracle_candidates(candidates, sample, cfg, oracle_mode="state_counterfactual")
    assert used == "state_counterfactual"
    best = pick_top(sc_oracle)
    assert best.action == true_reach
    assert best.action != legacy_b1.action

    # Verify legacy score function still works directly
    s_leg, _ = action_only_legacy_score(blind_lift, cfg)
    s_sc, out_sc = state_counterfactual_score(blind_lift, sample, cfg)
    assert s_leg > 0
    assert "lifted=no" in out_sc


def test_microduck_rollout_oracle_scene_aware() -> None:
    """When ball is already inside zone on conditional task, hold beats walking away."""
    sample = {
        "task": "if the orange ball is outside the circle then push it in else hold",
        "task_kind": "conditional",
        "scene_state": {
            "duck_x": -0.5,
            "duck_y": 0.0,
            "duck_yaw": 0.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
            "zone_cx": 0.0,
            "zone_cy": 0.0,
            "zone_r": 0.35,
        },
    }
    s_hold, out_hold = microduck_rollout_score([0.0, 0.0, 0.0], sample)
    s_walk, _ = microduck_rollout_score([0.25, 0.0, 0.5], sample)
    s_kick, _ = microduck_rollout_score([0.0, 0.0, 9.0], sample)
    assert "ball_in_zone=yes" in out_hold
    assert s_hold > s_walk
    assert s_hold > s_kick


def test_direct_policy_roundtrip_libero_and_microduck() -> None:
    lib_act = [0.045, -0.02, 0.015, 0.0, 0.0, 0.0, 1.0]
    txt = format_libero_policy_target(lib_act)
    parsed = parse_libero_policy_action(txt)
    assert parsed is not None
    assert action_distance(lib_act, parsed) < 1e-4

    duck_walk = [0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    d_txt = format_microduck_policy_target(duck_walk)
    d_parsed = parse_microduck_policy_action(d_txt)
    assert d_parsed == [0.25, 0.0, -0.5]

    duck_kick = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    k_txt = format_microduck_policy_target(duck_kick)
    k_parsed = parse_microduck_policy_action(k_txt)
    assert k_parsed == [0.0, 0.0, 9.0]


def test_robosuite_osc_mapping() -> None:
    osc_close = _to_robosuite_osc([0.05, -0.025, 0.0, 0.0, 0.0, 0.0, 1.0], osc_scale=10.0)
    assert abs(float(osc_close[0]) - 0.5) < 1e-5
    assert float(osc_close[6]) == 1.0
    osc_open = _to_robosuite_osc([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], osc_scale=10.0)
    assert float(osc_open[6]) == -1.0


def test_synthetic_tabletop_closed_loop(tmp_path: Path) -> None:
    env = SyntheticTabletopEnv(
        eef_x=0.0, eef_y=0.0, eef_z=0.02, obj_x=0.0, obj_y=0.0, obj_z=0.02
    )
    _, done1 = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # grasp
    assert env.grasped and not done1
    _, done2 = env.step([0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0])  # lift
    assert done2

    out_json = tmp_path / "closed_loop.json"
    controllers = {
        "B0_random": lambda _img, _task, rng: [
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.02, 0.04),
            0.0,
            0.0,
            0.0,
            float(rng.random() > 0.5),
        ],
    }
    res = run_synthetic_tabletop(
        controllers,
        n_tasks=2,
        episodes_per_task=2,
        max_steps=10,
        action_repeat=1,
        seed=42,
        out_path=out_json,
    )
    assert out_json.is_file()
    assert "B0_random" in res["macro_average"]


def test_microduck_planner_eval_config_and_evaluate_split(tmp_path: Path) -> None:
    import json
    import random

    from PIL import Image

    from swm.data.dataset import SemanticTransitionDataset
    from swm.eval.planner_eval import (
        domain_candidate_actions,
        evaluate_split,
        score_random,
    )
    from swm.utils.config import load_config

    md_cfg = load_config("configs/microduck_train.yaml")
    cands = domain_candidate_actions(md_cfg, "microduck")
    assert len(cands) == 8
    assert all(len(a) == 7 for a in cands)
    assert [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] in cands

    img_path = tmp_path / "f0.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img_path)
    split_file = tmp_path / "val.jsonl"
    row = {
        "episode_id": 0,
        "frame_idx": 0,
        "episode_len": 20,
        "task": "walk from zone A to zone B",
        "task_kind": "walk_ab",
        "action": [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "scene_state": {
            "duck_x": -0.5,
            "duck_y": 0.0,
            "duck_yaw": 0.0,
            "ball_x": 0.3,
            "ball_y": 0.1,
            "zone_cx": 0.0,
            "zone_cy": 0.0,
            "zone_r": 0.35,
            "goal_x": 1.2,
            "goal_y": 0.0,
            "zone_b_x": 0.5,
            "zone_b_y": 0.0,
        },
        "outcome": "moved=yes; lifted=no; gripper=hold; progress=up; success_near=no; ball_in_zone=no; facing_goal=yes; skill=walk",
        "image_path": str(img_path),
    }
    split_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = SemanticTransitionDataset(split_file, tmp_path)
    out = evaluate_split(
        ds,
        md_cfg,
        scorers={"B0_random": lambda acts, _i, _t: score_random(acts, random.Random(0))},
        max_samples=1,
        horizon=1,
        seed=42,
        domain="microduck",
        oracle_mode="microduck_rollout",
    )
    assert out["oracle_modes_used"] == ["microduck_rollout"]
    assert out["n_frames"] == 1

