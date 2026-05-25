#!/usr/bin/env python3
"""Evaluate train_red_ppo checkpoints in the clean CC4 tree.

This is a narrowed version of the parent repository's eval harness: it only
loads the trainable red PPO checkpoints produced by scripts/train/train_red_ppo.py
and evaluates them in the same wrapped CC4 environment used for training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_REPO_ROOT)

for path in (
    _REPO_ROOT,
    _REPO_ROOT / "on-policy",
    _REPO_ROOT / "Hierarchical-MARL-main" / "marl-1policy",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from scripts.train.train_red_ppo import CybORGRedMAPPOEnv, _build_onpolicy_args


NUM_RED_AGENTS = 6
ACTION_NAMES = {
    0: "DiscoverRemoteSystems",
    1: "StealthServiceDiscovery",
    2: "AggressiveServiceDiscovery",
    3: "ExploitRemoteService",
    4: "DegradeServices",
    5: "DiscoverDeception",
    6: "Withdraw",
    7: "PrivilegeEscalate",
    8: "Impact",
    9: "Sleep",
}


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_runner_meta(checkpoint_dir: Path) -> dict[str, Any]:
    meta_path = checkpoint_dir / "runner_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if (checkpoint_dir / "actor.pt").exists():
        return {"share_policy": True, "share_critic": False}
    if (checkpoint_dir / "actor_agent_0.pt").exists():
        return {"share_policy": False, "share_critic": False}
    raise FileNotFoundError(
        "No train_red_ppo actor checkpoint found in "
        f"{checkpoint_dir}. Expected actor.pt or actor_agent_0.pt."
    )


def _first_actor_path(checkpoint_dir: Path, share_policy: bool) -> Path:
    return checkpoint_dir / "actor.pt" if share_policy else checkpoint_dir / "actor_agent_0.pt"


def _infer_actor_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    hidden_size = 1024
    first_weight = state_dict.get("base.mlp.fc1.0.weight")
    if first_weight is not None:
        hidden_size = int(first_weight.shape[0])

    layer_indices = []
    prefix = "base.mlp.fc2."
    suffix = ".0.weight"
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(suffix):
            maybe_idx = key[len(prefix) : -len(suffix)]
            if maybe_idx.isdigit():
                layer_indices.append(int(maybe_idx))
    layer_n = max(layer_indices) + 1 if layer_indices else 2
    return hidden_size, layer_n


class RedPPOPolicyAdapter:
    """Loads train_red_ppo actor checkpoints and serves actions for eval."""

    def __init__(
        self,
        checkpoint_dir: Path,
        train_alg: str,
        eval_env: CybORGRedMAPPOEnv,
        device: torch.device,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.train_alg = train_alg
        self.eval_env = eval_env
        self.device = device
        self.meta = _load_runner_meta(checkpoint_dir)
        self.share_policy = bool(self.meta.get("share_policy", False))
        self.share_critic = bool(self.meta.get("share_critic", False))
        first_actor_state = torch.load(
            _first_actor_path(checkpoint_dir, self.share_policy),
            map_location=device,
        )
        hidden_size, layer_n = _infer_actor_architecture(first_actor_state)

        self.args = _build_onpolicy_args(
            train_alg=train_alg,
            num_iterations=1,
            num_rollout_threads=1,
            num_gpus=0,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=1,
            share_policy=self.share_policy,
            share_critic=self.share_critic,
        )
        self.args.cuda = device.type == "cuda"
        self.args.hidden_size = hidden_size
        self.args.layer_N = layer_n

        share_obs_space = (
            eval_env.share_observation_space[0]
            if self.args.use_centralized_V
            else eval_env.observation_space[0]
        )

        if self.share_policy:
            self.policy = R_MAPPOPolicy(
                self.args,
                eval_env.observation_space[0],
                share_obs_space,
                eval_env.action_space[0],
                device=device,
            )
            self.policy.actor.load_state_dict(first_actor_state)
            self.policy.actor.eval()
        else:
            self.policies = []
            for agent_idx in range(NUM_RED_AGENTS):
                actor_path = checkpoint_dir / f"actor_agent_{agent_idx}.pt"
                if not actor_path.exists():
                    raise FileNotFoundError(f"Missing per-agent actor checkpoint: {actor_path}")
                policy = R_MAPPOPolicy(
                    self.args,
                    eval_env.observation_space[0],
                    share_obs_space,
                    eval_env.action_space[0],
                    device=device,
                )
                state = first_actor_state if agent_idx == 0 else torch.load(actor_path, map_location=device)
                policy.actor.load_state_dict(state)
                policy.actor.eval()
                self.policies.append(policy)

        self.reset()

    def reset(self):
        recurrent_n = int(getattr(self.args, "recurrent_N", 1))
        hidden_size = int(getattr(self.args, "hidden_size", 1024))
        self.rnn_states = np.zeros(
            (NUM_RED_AGENTS, recurrent_n, hidden_size),
            dtype=np.float32,
        )

    @torch.no_grad()
    def act(self, obs, available_actions, deterministic: bool = True):
        masks = np.ones((NUM_RED_AGENTS, 1), dtype=np.float32)

        if self.share_policy:
            actions, rnn_states = self.policy.act(
                obs,
                self.rnn_states,
                masks,
                available_actions=available_actions,
                deterministic=deterministic,
            )
            self.rnn_states = rnn_states.detach().cpu().numpy()
            return actions.detach().cpu().numpy().astype(np.int64)

        actions = []
        next_rnn_states = []
        for agent_idx, policy in enumerate(self.policies):
            action, rnn_state = policy.act(
                obs[agent_idx : agent_idx + 1],
                self.rnn_states[agent_idx : agent_idx + 1],
                masks[agent_idx : agent_idx + 1],
                available_actions=available_actions[agent_idx : agent_idx + 1],
                deterministic=deterministic,
            )
            actions.append(action.detach().cpu().numpy()[0])
            next_rnn_states.append(rnn_state.detach().cpu().numpy()[0])

        self.rnn_states = np.stack(next_rnn_states, axis=0)
        return np.stack(actions, axis=0).astype(np.int64)


def _resolve_checkpoint_dir(args) -> Path:
    if args.red_policy_dir:
        return Path(args.red_policy_dir).expanduser().resolve()
    return (
        _REPO_ROOT
        / "red_checkpoints"
        / "ppo"
        / args.RL_TRAIN_ALG
        / args.red_policy_tag
    ).resolve()


def _configure_blue(args):
    os.environ["BLUE_OPPONENT"] = args.blue_opponent
    if args.blue_checkpoint_root:
        os.environ["BLUE_CHECKPOINT_ROOT"] = str(Path(args.blue_checkpoint_root).expanduser())
    if args.blue_checkpoint_iter:
        os.environ["BLUE_CHECKPOINT_ITER"] = args.blue_checkpoint_iter
    if args.blue_policy_dir:
        os.environ["BLUE_POLICY_DIR"] = str(Path(args.blue_policy_dir).expanduser())


def _run_episode(
    episode_idx: int,
    env: CybORGRedMAPPOEnv,
    policy: RedPPOPolicyAdapter,
    max_steps: int,
    deterministic: bool,
    keep_step_log: bool,
):
    obs = env.reset()
    policy.reset()

    total_reward = 0.0
    per_agent_reward = np.zeros(NUM_RED_AGENTS, dtype=np.float64)
    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    impact_attempts = 0
    successful_impacts = 0
    step_log = []

    for step_idx in range(max_steps):
        available_actions = env.available_actions.copy()
        active_before = env.active_masks.copy()
        actions = policy.act(
            obs,
            available_actions,
            deterministic=deterministic,
        )

        next_obs, rewards, dones, infos = env.step(actions)
        rewards_flat = rewards[:, 0].astype(np.float64)
        success_bits = next_obs[:, 1] > 0.5

        total_reward += float(rewards_flat.sum())
        per_agent_reward += rewards_flat

        for agent_idx, action in enumerate(actions):
            if active_before[agent_idx, 0] <= 0.0:
                continue
            action_type = int(action[0])
            action_name = ACTION_NAMES.get(action_type, f"Action{action_type}")
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            if action_type == 8:
                impact_attempts += 1
                if bool(success_bits[agent_idx]):
                    successful_impacts += 1

        if keep_step_log:
            for agent_idx, action in enumerate(actions):
                if active_before[agent_idx, 0] <= 0.0:
                    continue
                action_type = int(action[0])
                step_log.append(
                    {
                        "episode": episode_idx,
                        "step": step_idx,
                        "agent": f"red_agent_{agent_idx}",
                        "action_type": action_type,
                        "action_name": ACTION_NAMES.get(action_type, f"Action{action_type}"),
                        "target_subnet": int(action[1]),
                        "target_host": int(action[2]),
                        "success": bool(success_bits[agent_idx]),
                        "reward": float(rewards_flat[agent_idx]),
                    }
                )

        obs = next_obs
        if bool(np.all(dones)):
            break

    episode_steps = step_idx + 1
    return {
        "episode": episode_idx,
        "steps": episode_steps,
        "red_reward": total_reward,
        "per_agent_reward": {
            f"red_agent_{idx}": float(value) for idx, value in enumerate(per_agent_reward)
        },
        "impact_attempts": impact_attempts,
        "successful_impacts": successful_impacts,
        "action_counts": action_counts,
        "step_log": step_log,
    }


def evaluate(args):
    checkpoint_dir = _resolve_checkpoint_dir(args)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Red policy directory does not exist: {checkpoint_dir}")

    _configure_blue(args)
    os.environ["RL_TRAIN_ALG"] = args.RL_TRAIN_ALG

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "cpu"
    )

    env = CybORGRedMAPPOEnv(seed=args.seed)
    policy = RedPPOPolicyAdapter(
        checkpoint_dir=checkpoint_dir,
        train_alg=args.RL_TRAIN_ALG,
        eval_env=env,
        device=device,
    )

    episode_results = []
    red_log = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for episode_idx in range(args.episodes):
            result = _run_episode(
                episode_idx=episode_idx,
                env=env,
                policy=policy,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
                keep_step_log=not args.no_step_log,
            )
            red_log.extend(result.pop("step_log"))
            episode_results.append(result)
            print(
                f"episode {episode_idx + 1}/{args.episodes}: "
                f"red_reward={result['red_reward']:.2f}, "
                f"impact={result['successful_impacts']}/{result['impact_attempts']}, "
                f"steps={result['steps']}"
            )
    finally:
        env.close()

    rewards = np.array([ep["red_reward"] for ep in episode_results], dtype=np.float64)
    impact_attempts = int(sum(ep["impact_attempts"] for ep in episode_results))
    successful_impacts = int(sum(ep["successful_impacts"] for ep in episode_results))

    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    for ep in episode_results:
        for name, count in ep["action_counts"].items():
            action_counts[name] = action_counts.get(name, 0) + int(count)

    summary = {
        "started_at": started_at,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "red_policy_dir": str(checkpoint_dir),
        "train_alg": args.RL_TRAIN_ALG,
        "blue_opponent": args.blue_opponent,
        "deterministic": not args.stochastic,
        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "reward_std": float(rewards.std(ddof=0)) if rewards.size else 0.0,
        "reward_min": float(rewards.min()) if rewards.size else 0.0,
        "reward_max": float(rewards.max()) if rewards.size else 0.0,
        "impact_attempts": impact_attempts,
        "successful_impacts": successful_impacts,
        "impact_success_rate": (
            float(successful_impacts / impact_attempts) if impact_attempts else 0.0
        ),
        "action_counts": action_counts,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with (output_dir / "episodes.json").open("w", encoding="utf-8") as f:
        json.dump(episode_results, f, indent=2, default=_json_default)
    with (output_dir / "red_log.json").open("w", encoding="utf-8") as f:
        json.dump(red_log, f, indent=2, default=_json_default)

    summary_lines = [
        "CC4 red PPO evaluation",
        f"red_policy_dir: {checkpoint_dir}",
        f"train_alg: {args.RL_TRAIN_ALG}",
        f"blue_opponent: {args.blue_opponent}",
        f"episodes: {args.episodes}",
        f"reward_mean: {summary['reward_mean']:.3f}",
        f"reward_std: {summary['reward_std']:.3f}",
        f"successful_impacts: {successful_impacts}",
        f"impact_attempts: {impact_attempts}",
        f"impact_success_rate: {summary['impact_success_rate']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"wrote evaluation outputs to {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clean CC4 train_red_ppo checkpoints.")
    parser.add_argument(
        "--RL_TRAIN_ALG",
        default="ippo",
        choices=["ippo", "mappo"],
        help="Training algorithm family used for the red checkpoint.",
    )
    parser.add_argument(
        "--red-policy-dir",
        default=None,
        help="Checkpoint directory containing runner_meta.json and actor weights.",
    )
    parser.add_argument(
        "--red-policy-tag",
        default="best",
        choices=["best", "latest"],
        help="Checkpoint tag under red_checkpoints/ppo/<alg>/ when --red-policy-dir is omitted.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using argmax/mode.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument(
        "--blue-opponent",
        default=os.environ.get("BLUE_OPPONENT", "marl"),
        choices=["sleep", "marl"],
        help="Blue opponent during eval. 'marl' loads a trained blue checkpoint.",
    )
    parser.add_argument(
        "--blue-checkpoint-root",
        default=str(_REPO_ROOT / "Hierarchical-MARL-main" / "marl-1policy" / "models" / "train_marl"),
        help="Root containing blue iter_*/policies/Agent* checkpoints.",
    )
    parser.add_argument("--blue-checkpoint-iter", default=os.environ.get("BLUE_CHECKPOINT_ITER", "iter_199"))
    parser.add_argument("--blue-policy-dir", default=os.environ.get("BLUE_POLICY_DIR"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary.json, episodes.json, and red_log.json.",
    )
    parser.add_argument("--no-step-log", action="store_true", help="Do not write per-step red_log entries.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            _REPO_ROOT
            / "evaluation_output"
            / f"red_ppo_{args.RL_TRAIN_ALG}_{args.red_policy_tag}"
        )
    return args


if __name__ == "__main__":
    evaluate(parse_args())
