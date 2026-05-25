# ── scripts/ bootstrap: ensure repo root is on sys.path ────────────────────────
# Lets `from CybORG import ...`, `from onpolicy import ...`, and the
# `Hierarchical-MARL-main/marl-1policy/` modules resolve when this file is
# invoked directly from the repo root via `python scripts/<bucket>/<name>.py`.
# Sibling imports work
# automatically because Python adds the __main__ script's directory to
# sys.path. Dependency paths below are resolved from this file's repository
# root, so the script can be launched either directly or through scripts/launch.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import inspect
import logging
import warnings
import time
import json
import re
import argparse
import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BaseWrapper, RedFlatWrapper

# --- Path Setup for MARL Dependencies ---
MARL_DIR = str(_REPO_ROOT / "Hierarchical-MARL-main" / "marl-1policy")
if MARL_DIR not in sys.path:
    sys.path.append(MARL_DIR)

ON_POLICY_DIR = str(_REPO_ROOT / "on-policy")
if ON_POLICY_DIR not in sys.path:
    sys.path.append(ON_POLICY_DIR)
from onpolicy.utils.multi_discrete import MultiDiscrete as OnPolicyMultiDiscrete

if "PYTHONPATH" in os.environ:
    if MARL_DIR not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] += os.pathsep + MARL_DIR
else:
    os.environ["PYTHONPATH"] = MARL_DIR

try:
    from action_mask_model_CC4 import TorchActionMaskModel
    from Ray_BlueAgent import MARLAgent
    from EnterpriseMAE_CC4 import EnterpriseMAE
except ImportError as e:
    print(f"Could not import MARL dependencies: {e}")
    sys.exit(1)

# Register Custom Model
ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
torch, _ = try_import_torch()


# --- Non-OT Impact Redirect Constants & Utility ---
# Impact only succeeds on OZA/OZB (subnets with OTSERVICE).  All other 7 subnets
# have 0% Impact success rate (env hard constraint: Impact.py checks for
# ProcessName.OTSERVICE, only operational_zone_* hosts have it).
OZ_SUBNET_INDICES = {1, 3}   # operational_zone_a_subnet, operational_zone_b_subnet
IMPACT_ACTION_TYPE = 8
DRS_ACTION_TYPE = 0           # DiscoverRemoteSystems (subnet-level)


def redirect_non_ot_impact(action):
    """Redirect Impact on non-OT subnets to DiscoverRemoteSystems targeting a random OT zone.

    RL agents waste 60-87% of Impact attempts on non-operational subnets where
    no OTSERVICE exists (guaranteed 0% success).  This redirects those structurally
    doomed actions to useful exploration of OT subnets.
    """
    act = np.asarray(action)
    if len(act) >= 2 and int(act[0]) == IMPACT_ACTION_TYPE:
        target_subnet = int(act[1])
        if target_subnet not in OZ_SUBNET_INDICES:
            oz_target = np.random.choice(list(OZ_SUBNET_INDICES))
            return np.array([DRS_ACTION_TYPE, oz_target, 0], dtype=act.dtype)
    return action


def _extract_action_subspace(space):
    """Use the factorized red action branch when wrappers expose action+message Dicts."""
    if hasattr(space, "spaces") and "action" in space.spaces:
        return space.spaces["action"]
    return space


class TorchConditionalMultiActionDist(TorchDistributionWrapper):
    """
    Autoregressive factorization for MultiDiscrete([action_type, target_subnet, target_host]):
    pi(a, s, h) = pi(a) * pi(s | a) * pi(h | a, s)
    """

    HOST_ACTION_TYPES = {1, 2, 3, 4, 5, 6, 7, 8}
    SUBNET_ACTION_TYPE = 0
    NO_TARGET_ACTION_TYPE = 9
    MASKED_LOGIT = -1e9

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        nvec = model.action_space.nvec
        self.n_action = int(nvec[0])
        self.n_subnet = int(nvec[1])
        self.n_host = int(nvec[2])
        self.action_logits, self.subnet_logits, self.host_logits = torch.split(
            self.inputs, [self.n_action, self.n_subnet, self.n_host], dim=1
        )
        self._mask_action = None
        self._mask_subnet = None
        self._mask_host = None
        self._mask_subnet_by_action = None
        self._mask_host_by_action_subnet = None
        self._load_model_masks(model)
        self._action_logp = None

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return int(np.sum(action_space.nvec))

    def _load_model_masks(self, model):
        action_mask = getattr(model, "_last_action_mask", None)
        if not isinstance(action_mask, dict):
            return
        self._mask_action = action_mask.get("action_type")
        self._mask_subnet = action_mask.get("target_subnet")
        self._mask_host = action_mask.get("target_host")
        self._mask_subnet_by_action = action_mask.get("target_subnet_by_action")
        self._mask_host_by_action_subnet = action_mask.get("target_host_by_action_subnet")

    def _safe_categorical(self, logits):
        safe_logits = logits.clone()
        invalid_rows = ~torch.isfinite(safe_logits).any(dim=1)
        if invalid_rows.any():
            safe_logits[invalid_rows] = self.MASKED_LOGIT
            safe_logits[invalid_rows, 0] = 0.0
        return torch.distributions.categorical.Categorical(logits=safe_logits)

    def _masked_action_logits(self):
        return self._mask_logits(self.action_logits, self._mask_action)

    def _mask_logits(self, logits, mask):
        if mask is None:
            return logits
        masked = logits.clone()
        mask = mask.to(device=masked.device, dtype=masked.dtype)
        masked[mask <= 0.0] = self.MASKED_LOGIT
        invalid_rows = (mask > 0.0).sum(dim=1) == 0
        if invalid_rows.any():
            masked[invalid_rows] = self.MASKED_LOGIT
            masked[invalid_rows, 0] = 0.0
        return masked

    def _gather_subnet_mask(self, action_types):
        if self._mask_subnet_by_action is None:
            return None
        mask = self._mask_subnet_by_action
        if mask.dim() != 3:
            return None
        action_types = torch.clamp(action_types.long(), 0, self.n_action - 1)
        index = action_types.view(-1, 1, 1).expand(-1, 1, self.n_subnet)
        return torch.gather(mask, 1, index).squeeze(1)

    def _gather_host_mask(self, action_types, target_subnets):
        if self._mask_host_by_action_subnet is None:
            return None
        mask = self._mask_host_by_action_subnet
        if mask.dim() != 4:
            return None

        action_types = torch.clamp(action_types.long(), 0, self.n_action - 1)
        target_subnets = torch.clamp(target_subnets.long(), 0, self.n_subnet - 1)

        action_index = action_types.view(-1, 1, 1, 1).expand(-1, 1, self.n_subnet, self.n_host)
        action_slice = torch.gather(mask, 1, action_index).squeeze(1)
        subnet_index = target_subnets.view(-1, 1, 1).expand(-1, 1, self.n_host)
        return torch.gather(action_slice, 1, subnet_index).squeeze(1)

    def _conditional_subnet_logits(self, action_types):
        subnet_logits = self._mask_logits(self.subnet_logits, self._mask_subnet)
        subnet_mask = self._gather_subnet_mask(action_types)
        subnet_logits = self._mask_logits(subnet_logits, subnet_mask)

        no_target_action = action_types == self.NO_TARGET_ACTION_TYPE
        if no_target_action.any():
            subnet_logits[no_target_action] = self.MASKED_LOGIT
            subnet_logits[no_target_action, 0] = 0.0

        return subnet_logits

    def _conditional_host_logits(self, action_types, target_subnets):
        host_logits = self._mask_logits(self.host_logits, self._mask_host)
        host_mask = self._gather_host_mask(action_types, target_subnets)
        host_logits = self._mask_logits(host_logits, host_mask)

        subnet_action = action_types == self.SUBNET_ACTION_TYPE
        no_target_action = action_types == self.NO_TARGET_ACTION_TYPE
        host_actions = torch.zeros_like(action_types, dtype=torch.bool)
        for aid in self.HOST_ACTION_TYPES:
            host_actions |= action_types == aid

        # Host-targeted actions require host_idx > 0.
        host_logits[host_actions, 0] = self.MASKED_LOGIT

        # Subnet-targeted and no-target actions deterministically use host_idx == 0.
        if subnet_action.any():
            host_logits[subnet_action] = self.MASKED_LOGIT
            host_logits[subnet_action, 0] = 0.0
        if no_target_action.any():
            host_logits[no_target_action] = self.MASKED_LOGIT
            host_logits[no_target_action, 0] = 0.0

        return host_logits

    def sample(self):
        action_dist = self._safe_categorical(self._masked_action_logits())
        action_type = action_dist.sample()
        subnet_logits = self._conditional_subnet_logits(action_type)
        subnet_dist = self._safe_categorical(subnet_logits)
        target_subnet = subnet_dist.sample()
        host_logits = self._conditional_host_logits(action_type, target_subnet)
        host_dist = self._safe_categorical(host_logits)
        target_host = host_dist.sample()

        self.last_sample = torch.stack([action_type, target_subnet, target_host], dim=1)
        self._action_logp = (
            action_dist.log_prob(action_type)
            + subnet_dist.log_prob(target_subnet)
            + host_dist.log_prob(target_host)
        )
        return self.last_sample

    def deterministic_sample(self):
        action_dist = self._safe_categorical(self._masked_action_logits())
        action_type = torch.argmax(action_dist.probs, dim=1)
        subnet_logits = self._conditional_subnet_logits(action_type)
        subnet_dist = self._safe_categorical(subnet_logits)
        target_subnet = torch.argmax(subnet_dist.probs, dim=1)
        host_logits = self._conditional_host_logits(action_type, target_subnet)
        host_dist = self._safe_categorical(host_logits)
        target_host = torch.argmax(host_dist.probs, dim=1)

        self.last_sample = torch.stack([action_type, target_subnet, target_host], dim=1)
        self._action_logp = (
            action_dist.log_prob(action_type)
            + subnet_dist.log_prob(target_subnet)
            + host_dist.log_prob(target_host)
        )
        return self.last_sample

    def logp(self, actions):
        if actions.dtype != torch.long:
            actions = actions.long()
        action_type = actions[:, 0]
        target_subnet = actions[:, 1]
        target_host = actions[:, 2]

        action_dist = self._safe_categorical(self._masked_action_logits())
        subnet_logits = self._conditional_subnet_logits(action_type)
        subnet_dist = self._safe_categorical(subnet_logits)
        host_logits = self._conditional_host_logits(action_type, target_subnet)
        host_dist = self._safe_categorical(host_logits)

        return (
            action_dist.log_prob(action_type)
            + subnet_dist.log_prob(target_subnet)
            + host_dist.log_prob(target_host)
        )

    def sampled_action_logp(self):
        if self._action_logp is None and self.last_sample is not None:
            return self.logp(self.last_sample)
        return self._action_logp

    def entropy(self):
        action_dist = self._safe_categorical(self._masked_action_logits())
        action_entropy = action_dist.entropy()
        action_probs = action_dist.probs

        target_entropy = torch.zeros_like(action_entropy)
        batch_size = self.action_logits.shape[0]
        for aid in range(self.n_action):
            aid_tensor = torch.full(
                (batch_size,),
                fill_value=aid,
                dtype=torch.long,
                device=self.action_logits.device,
            )
            subnet_logits = self._conditional_subnet_logits(aid_tensor)
            subnet_dist = self._safe_categorical(subnet_logits)
            subnet_entropy = subnet_dist.entropy()
            subnet_probs = subnet_dist.probs

            host_entropy_expectation = torch.zeros_like(subnet_entropy)
            for sid in range(self.n_subnet):
                sid_tensor = torch.full(
                    (batch_size,),
                    fill_value=sid,
                    dtype=torch.long,
                    device=self.action_logits.device,
                )
                host_logits = self._conditional_host_logits(aid_tensor, sid_tensor)
                host_entropy_sid = self._safe_categorical(host_logits).entropy()
                host_entropy_expectation += subnet_probs[:, sid] * host_entropy_sid

            target_entropy += action_probs[:, aid] * (subnet_entropy + host_entropy_expectation)
        return action_entropy + target_entropy

    def kl(self, other):
        action_dist = self._safe_categorical(self._masked_action_logits())
        other_action_dist = other._safe_categorical(other._masked_action_logits())
        action_kl = torch.distributions.kl.kl_divergence(action_dist, other_action_dist)
        action_probs = action_dist.probs

        target_kl = torch.zeros_like(action_kl)
        batch_size = self.action_logits.shape[0]
        for aid in range(self.n_action):
            aid_tensor = torch.full(
                (batch_size,),
                fill_value=aid,
                dtype=torch.long,
                device=self.action_logits.device,
            )
            subnet_logits = self._conditional_subnet_logits(aid_tensor)
            other_subnet_logits = other._conditional_subnet_logits(aid_tensor)
            subnet_dist = self._safe_categorical(subnet_logits)
            subnet_probs = subnet_dist.probs
            subnet_kl = torch.distributions.kl.kl_divergence(
                subnet_dist,
                self._safe_categorical(other_subnet_logits),
            )

            host_kl_expectation = torch.zeros_like(subnet_kl)
            for sid in range(self.n_subnet):
                sid_tensor = torch.full(
                    (batch_size,),
                    fill_value=sid,
                    dtype=torch.long,
                    device=self.action_logits.device,
                )
                host_logits = self._conditional_host_logits(aid_tensor, sid_tensor)
                other_host_logits = other._conditional_host_logits(aid_tensor, sid_tensor)
                host_kl_sid = torch.distributions.kl.kl_divergence(
                    self._safe_categorical(host_logits),
                    self._safe_categorical(other_host_logits),
                )
                host_kl_expectation += subnet_probs[:, sid] * host_kl_sid

            target_kl += action_probs[:, aid] * (subnet_kl + host_kl_expectation)
        return action_kl + target_kl


ModelCatalog.register_custom_action_dist("red_conditional_dist", TorchConditionalMultiActionDist)


# Penalty applied to a red agent when it chooses Impact (action type 8) but the


class CustomPwnRewardCalculator:
    def __init__(self, agent_name, scenario_generator):
        self.agent_name = agent_name
        self.scenario_generator = scenario_generator

    def calculate_simulation_reward(self, env_controller):
        """
        Calculates the Pwn reward based on the current state.
        Reward = Sum of values of compromised hosts.
        Root access = 10.0
        User access = 1.0
        """
        score = 0.0
        state = env_controller.state
        
        # We need to check all hosts
        for hostname, host in state.hosts.items():
            if 'router' in hostname: continue # Ignore routers
            
            # Check for Red sessions
            has_root = False
            has_user = False
            
            # Iterate over all red agents
            for agent in state.sessions.keys():
                if 'red' in agent:
                    sessions = host.sessions.get(agent, [])
                    for session_id in sessions:
                        session = state.sessions[agent][session_id]
                        if session.username == 'root' or session.username == 'SYSTEM':
                            has_root = True
                        else:
                            has_user = True
            
            if has_root:
                score += 10.0
            elif has_user:
                score += 1.0
                
        return score

class RedRewardWrapper(BaseWrapper):
    """
    Shapes Red reward using simulation-state progress.
    Reward contribution is the delta of Pwn score between consecutive steps.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_calculator = CustomPwnRewardCalculator("Red", env.scenario_generator)
        self.previous_score = 0.0

    def reset(self, **kwargs):
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        self.previous_score = self.reward_calculator.calculate_simulation_reward(
            self.env.environment_controller
        )
        return obs, info

    def step(self, action=None, messages=None, **kwargs):
        obs, rewards, terminated, truncated, info = self.env.step(action, messages, **kwargs)

        current_score = self.reward_calculator.calculate_simulation_reward(
            self.env.environment_controller
        )
        pwn_delta = float(current_score - self.previous_score)
        self.previous_score = current_score

        for agent_name in list(rewards.keys()):
            if "red" in agent_name:
                rewards[agent_name] = float(rewards.get(agent_name, 0.0)) + pwn_delta

        return obs, rewards, terminated, truncated, info


class HybridFixedBlueWrapper(BaseWrapper):
    """
    Wraps the environment to:
    1. Manage Blue Agents (MARLAgent) internally.
    2. Provide EnterpriseMAE-formatted observations to Blue Agents.
    3. Provide Raw/Red-formatted observations to Red Agents (passing through to child wrappers).
    4. Inject Blue Actions into the step.
    """
    def __init__(self, env, blue_opponent=None):
        # We wrap the Blue-configured environment (EnterpriseMAE)
        super().__init__(env)

        self.blue_opponent = (blue_opponent or os.environ.get("BLUE_OPPONENT", "sleep")).lower()
        if self.blue_opponent == "marl":
            # Use a trained blue checkpoint. See BLUE_CHECKPOINT_* env vars in
            # Ray_BlueAgent.py for checkpoint path selection.
            self.blue_agents = {
                f"blue_agent_{i}": MARLAgent(name=f"Agent{i}") for i in range(5)
            }
        elif self.blue_opponent == "sleep":
            # Leave blue actions to the underlying CybORG SleepAgent defaults.
            self.blue_agents = {}
        else:
            raise ValueError("BLUE_OPPONENT must be either 'sleep' or 'marl'")
        
        # We wrap env with EnterpriseMAE to handle Blue state/actions
        self.blue_env_wrapper = EnterpriseMAE(env) 
        
        self.blue_observations = {}

    def _sum_reward_components(self, reward_obj):
        if isinstance(reward_obj, dict):
            return float(sum(reward_obj.values()))
        if reward_obj is None:
            return 0.0
        try:
            return float(reward_obj)
        except (TypeError, ValueError):
            return 0.0

    def _normalize_allowed_subnets(self, allowed_subnets):
        normalized = set()
        for subnet in allowed_subnets or []:
            subnet_name = getattr(subnet, "value", subnet)
            normalized.add(str(subnet_name))
        return normalized

    def _active_red_reward_by_blue(self, cyborg, blue_rewards):
        """
        Assign each active red agent a dense baseline reward derived from Blue's
        per-agent step rewards:
        - If red and blue allowed-subnet sets overlap: use the negative mean of overlapping blue rewards.
        - If no overlap exists (e.g., contractor red): fall back to negative mean over all blue rewards.
        """
        if not blue_rewards:
            return {}

        interfaces = cyborg.environment_controller.agent_interfaces
        blue_allowed_subnets = {}
        for agent_name in blue_rewards:
            interface = interfaces.get(agent_name)
            if interface is None:
                blue_allowed_subnets[agent_name] = set()
                continue
            blue_allowed_subnets[agent_name] = self._normalize_allowed_subnets(
                getattr(interface, "allowed_subnets", [])
            )

        blue_default = float(np.mean(list(blue_rewards.values())))
        red_rewards = {}
        for agent_name in cyborg.agents:
            if "red" not in agent_name:
                continue
            interface = interfaces.get(agent_name)
            if interface is None or not interface.active:
                continue

            red_allowed = self._normalize_allowed_subnets(
                getattr(interface, "allowed_subnets", [])
            )
            overlapping_blue = [
                blue_agent
                for blue_agent, blue_subnets in blue_allowed_subnets.items()
                if red_allowed & blue_subnets
            ]
            if overlapping_blue:
                mapped_blue_reward = float(
                    np.mean([blue_rewards[blue_agent] for blue_agent in overlapping_blue])
                )
            else:
                mapped_blue_reward = blue_default

            red_rewards[agent_name] = -mapped_blue_reward
        return red_rewards

    def reset(self, **kwargs):
        # Reset the Blue Wrapper (which resets the underlying env)
        blue_obs, info = self.blue_env_wrapper.reset(**kwargs)
        
        # Store Blue Obs for first step
        self.blue_observations = blue_obs
        
        # Recover Red Obs
        red_obs = {}
        # Access the raw environment controller
        cyborg = self.blue_env_wrapper.env 
        
        for agent in cyborg.agents:
            if 'red' in agent and cyborg.environment_controller.agent_interfaces[agent].active:
                red_obs[agent] = cyborg.get_observation(agent)
                
        return red_obs, info

    def step(self, action_dict, messages=None, **kwargs):
        # 1. Get Blue Actions
        blue_actions = {}
        for agent_name, agent_obj in self.blue_agents.items():
            if agent_name in self.blue_observations:
                action = agent_obj.get_action(self.blue_observations[agent_name], None)
                blue_actions[agent_name] = action

        # 2. Combine Actions
        all_actions = {**action_dict, **blue_actions}
        
        # 3. Step the Blue Wrapper
        b_obs, b_rew, b_term, b_trunc, b_info = self.blue_env_wrapper.step(all_actions, messages=messages, **kwargs)

        # Store Blue Obs for next step
        self.blue_observations = b_obs

        # 3.5. Build Blue per-agent rewards and a summary total.
        blue_rewards = {
            agent_name: self._sum_reward_components(rew)
            for agent_name, rew in b_rew.items()
            if "blue" in agent_name
        }
        blue_total_reward = float(sum(blue_rewards.values()))
        
        # 4. Recover Red Obs & Rewards
        cyborg = self.blue_env_wrapper.env
        red_rewards = self._active_red_reward_by_blue(cyborg, blue_rewards)
        
        red_obs = {}
        red_rew = {}
        red_term = {}
        red_trunc = {}
        
        for agent in cyborg.agents:
            if 'red' in agent:
                if cyborg.environment_controller.agent_interfaces[agent].active:
                    # Use get_observation() so received messages (from send_messages()) are
                    # included; get_agent_state() uses get_true_state() which bypasses the
                    # observation cache and strips "message" entries.
                    raw_obs = cyborg.get_observation(agent)
                    red_obs[agent] = raw_obs
                    red_term[agent] = False
                    red_trunc[agent] = False
                    # Pure blue-derived reward signal (negative of mapped blue reward).
                    red_rew[agent] = red_rewards.get(agent, 0.0)

        # Global done
        red_term["__all__"] = b_term.get("__all__", False)
        red_trunc["__all__"] = b_trunc.get("__all__", False)

        # Expose Blue total reward to downstream wrappers (e.g., RedRewardWrapper).
        if isinstance(b_info, dict):
            b_info["__blue_total_reward"] = blue_total_reward
            b_info["__blue_rewards"] = blue_rewards

        return red_obs, red_rew, red_term, red_trunc, b_info

    def action_space(self, agent):
        if 'red' in agent:
            cyborg = self.blue_env_wrapper.env
            return cyborg.action_space(agent)
        return self.blue_env_wrapper.action_space(agent)

    def observation_space(self, agent):
        if 'red' in agent:
            cyborg = self.blue_env_wrapper.env
            return cyborg.observation_space(agent)
        return self.blue_env_wrapper.observation_space(agent)
    
    def get_action_space(self, agent):
        return self.action_space(agent)
        
    def get_observation_space(self, agent):
        return self.observation_space(agent)


class RedActionMaskWrapper(BaseWrapper):
    """
    Combines observation and action mask into a Dict space for RLLib.
    """
    def reset(self, **kwargs):
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        return self._add_mask(obs, info), info

    def step(self, action=None, messages=None, **kwargs):
        # Redirect Impact on non-OT subnets → DiscoverRemoteSystems (OZA/OZB).
        # At this level, actions are still raw factorized numpy arrays.
        if action is not None:
            action = {agent: redirect_non_ot_impact(a) for agent, a in action.items()}
        obs, rewards, terminated, truncated, info = self.env.step(action, messages, **kwargs)
        return self._add_mask(obs, info), rewards, terminated, truncated, info

    def _add_mask(self, obs, info):
        new_obs = {}
        for agent, o in obs.items():
            if 'red' not in agent:
                continue

            act_space = _extract_action_subspace(self.env.action_space(agent))
            n_action = int(act_space.nvec[0])
            n_subnet = int(act_space.nvec[1])
            n_host = int(act_space.nvec[2])

            mask = None
            if agent in info and 'action_mask' in info[agent]:
                mask = info[agent]['action_mask']

            if mask is None:
                mask = {
                    "action_type": np.ones(n_action, dtype=np.float32),
                    "target_subnet": np.ones(n_subnet, dtype=np.float32),
                    "target_host": np.ones(n_host, dtype=np.float32),
                    "target_subnet_by_action": np.ones((n_action, n_subnet), dtype=np.float32),
                    "target_host_by_action_subnet": np.ones((n_action, n_subnet, n_host), dtype=np.float32),
                }
            else:
                subnet_by_action = mask.get("target_subnet_by_action")
                if subnet_by_action is None:
                    subnet_by_action = np.repeat(
                        np.array(mask["target_subnet"], dtype=np.float32)[None, :],
                        n_action,
                        axis=0,
                    )

                host_by_action_subnet = mask.get("target_host_by_action_subnet")
                if host_by_action_subnet is None:
                    host_by_action_subnet = np.repeat(
                        np.array(mask["target_host"], dtype=np.float32)[None, None, :],
                        n_action,
                        axis=0,
                    )
                    host_by_action_subnet = np.repeat(host_by_action_subnet, n_subnet, axis=1)

                mask = {
                    "action_type": np.array(mask["action_type"], dtype=np.float32),
                    "target_subnet": np.array(mask["target_subnet"], dtype=np.float32),
                    "target_host": np.array(mask["target_host"], dtype=np.float32),
                    "target_subnet_by_action": np.array(subnet_by_action, dtype=np.float32),
                    "target_host_by_action_subnet": np.array(host_by_action_subnet, dtype=np.float32),
                }

            new_obs[agent] = {
                "observations": o,
                "action_mask": mask,
            }
        return new_obs
        
    def observation_space(self, agent):
        orig_space = self.env.observation_space(agent)
        act_space = _extract_action_subspace(self.env.action_space(agent))
        return spaces.Dict({
            "observations": orig_space,
            "action_mask": spaces.Dict({
                "action_type": spaces.Box(0.0, 1.0, shape=(act_space.nvec[0],), dtype=np.float32),
                "target_subnet": spaces.Box(0.0, 1.0, shape=(act_space.nvec[1],), dtype=np.float32),
                "target_host": spaces.Box(0.0, 1.0, shape=(act_space.nvec[2],), dtype=np.float32),
                "target_subnet_by_action": spaces.Box(
                    0.0,
                    1.0,
                    shape=(act_space.nvec[0], act_space.nvec[1]),
                    dtype=np.float32,
                ),
                "target_host_by_action_subnet": spaces.Box(
                    0.0,
                    1.0,
                    shape=(act_space.nvec[0], act_space.nvec[1], act_space.nvec[2]),
                    dtype=np.float32,
                ),
            })
        })

    def action_space(self, agent):
        return _extract_action_subspace(self.env.action_space(agent))


class RedMAERayWrapper(MultiAgentEnv):
    """
    RLLib compatible wrapper for CybORG Red Agent.
    """
    def __init__(self, env):
        self.env = env
        self.agents = [] 
        self._agent_ids = set()
        super().__init__()

    def reset(self, *, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
        obs, info = self.env.reset(**kwargs)
        obs_keys = set(obs.keys())
        if isinstance(info, dict):
            info = {
                agent_id: payload
                for agent_id, payload in info.items()
                if agent_id in obs_keys or agent_id == "__common__"
            }
        else:
            info = {}
        self.agents = list(obs.keys())
        self._agent_ids = set(self.agents)
        return obs, info

    def step(self, action_dict):
        obs, rews, terms, truncs, infos = self.env.step(action_dict)
        obs_keys = set(obs.keys())

        if isinstance(infos, dict):
            infos = {
                agent_id: payload
                for agent_id, payload in infos.items()
                if agent_id in obs_keys or agent_id == "__common__"
            }
        else:
            infos = {}

        if "__all__" not in terms:
            terms["__all__"] = all(terms.values())
        if "__all__" not in truncs:
            truncs["__all__"] = all(truncs.values())
            
        self.agents = list(obs.keys())
        return obs, rews, terms, truncs, infos

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)


def env_creator(env_config):
    # Imports are now at top level

    # 1. Create Base CybORG with SleepAgent for Blue 
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    
    # 2. Hybrid Wrapper (Injects Blue Agents + Blue Obs)
    cyborg = HybridFixedBlueWrapper(cyborg)
    
    # 3. Red Flattening (Obs Vectorization)
    cyborg = RedFlatWrapper(cyborg)

    # 4. Action Masking
    cyborg = RedActionMaskWrapper(cyborg)
    
    # 5. RLLib MultiAgent Adapter
    env = RedMAERayWrapper(cyborg)
    
    return env


register_env("CybORG_Red_PPO", env_creator)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # We will pass the algorithm name via env runner config if needed, 
    # but for now we can use a global variable or env var since it's set at startup.
    train_alg = os.environ.get("RL_TRAIN_ALG", "ippo").lower()
    if train_alg == "ippo":
        if "red" in agent_id:
            return f"{agent_id}_policy"
    return "red_policy"


def setup_training_logger(log_path):
    logger = logging.getLogger("train_red_ppo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _to_finite_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _first_metric(candidates, aliases):
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in aliases:
            if key in candidate:
                value = _to_finite_float(candidate[key])
                if value is not None:
                    return value
    return None


def extract_optimization_metrics(result):
    candidates = []

    learners = result.get("learners")
    if isinstance(learners, dict):
        # Collect metrics from all red policies
        red_policy_ids = [pid for pid in learners.keys() if "red" in pid and "policy" in pid]
        for pid in red_policy_ids:
            red_policy_metrics = learners.get(pid)
            if isinstance(red_policy_metrics, dict):
                candidates.append(red_policy_metrics)
                learner_stats = red_policy_metrics.get("learner_stats")
                if isinstance(learner_stats, dict):
                    candidates.append(learner_stats)

    info = result.get("info")
    if isinstance(info, dict):
        learner_info = info.get("learner")
        if isinstance(learner_info, dict):
            red_policy_ids = [pid for pid in learner_info.keys() if "red" in pid and "policy" in pid]
            for pid in red_policy_ids:
                red_policy_info = learner_info.get(pid)
                if isinstance(red_policy_info, dict):
                    candidates.append(red_policy_info)
                    learner_stats = red_policy_info.get("learner_stats")
                    if isinstance(learner_stats, dict):
                        candidates.append(learner_stats)

    alias_map = {
        "total_loss": ["total_loss", "loss"],
        "policy_loss": ["policy_loss", "mean_policy_loss", "surrogate_loss"],
        "value_loss": ["vf_loss", "value_loss", "mean_vf_loss", "vf_loss_unclipped"],
        "entropy": ["entropy", "mean_entropy"],
        "kl": ["mean_kl_loss", "kl", "kl_loss"],
        "explained_var": ["vf_explained_var", "explained_variance"],
        "learning_rate": ["cur_lr", "default_optimizer_lr", "lr"],
        "entropy_coeff": ["curr_entropy_coeff", "entropy_coeff"],
        "kl_coeff": ["curr_kl_coeff", "kl_coeff"],
        "grad_norm": ["grad_gnorm", "grad_norm"],
    }

    metrics = {}
    for metric_name, aliases in alias_map.items():
        value = _first_metric(candidates, aliases)
        if value is not None:
            metrics[metric_name] = value
    return metrics


def _format_seconds(seconds):
    if seconds is None:
        return "n/a"
    seconds = max(float(seconds), 0.0)
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    return f"{mins:d}m{secs:02d}s"


def _resolve_ray_num_cpus():
    override = os.environ.get("RL_RAY_NUM_CPUS")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid RL_RAY_NUM_CPUS=%s, ignore override.", override)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            value = int(slurm_cpus)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid SLURM_CPUS_PER_TASK=%s, fallback to os.cpu_count().", slurm_cpus)

    return max(1, int(os.cpu_count() or 1))


def _is_better_metric(current_value, best_value, mode):
    if current_value is None:
        return False
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value
    return current_value > best_value


def _resolve_best_metric_value(metric_name, reward_mean, opt_metrics, result):
    if metric_name in {"reward_mean", "episode_reward_mean"}:
        return _to_finite_float(reward_mean)
    if metric_name.startswith("opt/"):
        opt_key = metric_name.split("/", 1)[1]
        return _to_finite_float(opt_metrics.get(opt_key))
    if metric_name in opt_metrics:
        return _to_finite_float(opt_metrics.get(metric_name))

    value = result.get(metric_name)
    if value is None and isinstance(result.get("env_runners"), dict):
        value = result["env_runners"].get(metric_name)
    return _to_finite_float(value)


def _checkpoint_result_path(save_result):
    if save_result is None:
        return None
    if isinstance(save_result, str):
        return save_result
    checkpoint = getattr(save_result, "checkpoint", None)

    for obj in (checkpoint, save_result):
        if obj is None:
            continue

        path = getattr(obj, "path", None)
        if path:
            return str(path)

        uri = getattr(obj, "uri", None)
        if isinstance(uri, str) and uri:
            return uri

        if isinstance(obj, dict):
            for key in ("path", "checkpoint_path", "uri"):
                value = obj.get(key)
                if isinstance(value, str) and value:
                    return value

        # Fallback for repr forms like:
        # TrainingResult(checkpoint=Checkpoint(filesystem=local, path=./red_checkpoints/ppo/ippo/latest), ...)
        text = repr(obj)
        match = re.search(r"path=([^,\)]+)", text)
        if match:
            parsed = match.group(1).strip().strip("'\"")
            if parsed and parsed.lower() != "none":
                return parsed

    return None


class CybORGRedMAPPOEnv:
    """
    On-policy style single-env adapter:
    - reset() -> obs[num_agents, obs_dim]
    - step(actions[num_agents, action_dim]) -> obs, rewards[num_agents,1], dones[num_agents], infos[list]
    """

    # Action type constants for structural masking
    SUBNET_ACTION_TYPE = 0        # DiscoverRemoteSystems
    HOST_ACTION_TYPES = {1, 2, 3, 4, 5, 6, 7, 8}
    NO_TARGET_ACTION_TYPE = 9     # Sleep

    def __init__(self, seed=None):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            steps=500,
        )
        cyborg = CybORG(scenario_generator=sg)
        cyborg = HybridFixedBlueWrapper(cyborg)
        cyborg = RedFlatWrapper(cyborg)
        self.env = cyborg
        self.seed = seed
        self.red_agents = [f"red_agent_{i}" for i in range(6)]
        self._active_agents = set()

        first_agent = self.red_agents[0]
        self._obs_space = self.env.observation_space(first_agent)
        self._act_space = _extract_action_subspace(self.env.action_space(first_agent))
        if self._act_space.__class__.__name__ == "MultiDiscrete":
            bounds = [[0, int(n) - 1] for n in self._act_space.nvec]
            self._policy_act_space = OnPolicyMultiDiscrete(bounds)
        else:
            self._policy_act_space = self._act_space

        # Compute hierarchical mask dimensions
        if hasattr(self._act_space, 'nvec'):
            self._nvec = [int(n) for n in self._act_space.nvec]
        else:
            self._nvec = [int(n) for n in (self._policy_act_space.high - self._policy_act_space.low + 1)]
        n_a, n_s, n_h = self._nvec[0], self._nvec[1], self._nvec[2]
        self._mask_size = n_a + n_a * n_s + n_a * n_s * n_h
        self._default_mask = self._make_default_mask(n_a, n_s, n_h)
        self._inactive_mask = self._make_sleep_only_mask(n_a, n_s, n_h)
        self.available_actions = np.tile(self._inactive_mask, (len(self.red_agents), 1))
        self.active_masks = np.zeros((len(self.red_agents), 1), dtype=np.float32)

        self.observation_space = [self._obs_space for _ in self.red_agents]
        # CTDE: MAPPO's centralized critic sees all red agents' observations
        # concatenated. Actors still use only their own local observation.
        share_dim = int(np.prod(self._obs_space.shape)) * len(self.red_agents)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32)
            for _ in self.red_agents
        ]
        self.action_space = [self._policy_act_space for _ in self.red_agents]
        self._zero_obs = np.zeros(self._obs_space.shape, dtype=np.float32)

    @staticmethod
    def _make_default_mask(n_a, n_s, n_h):
        """Build an all-ones hierarchical mask with structural constraints baked in."""
        action_type_mask = np.ones(n_a, dtype=np.float32)
        subnet_by_action = np.ones((n_a, n_s), dtype=np.float32)
        host_by_action_subnet = np.ones((n_a, n_s, n_h), dtype=np.float32)
        # Apply structural constraints
        # Subnet-targeted (0): host must be 0
        host_by_action_subnet[0, :, 1:] = 0.0
        # Host-targeted (1-8): host must be > 0
        for a in range(1, min(9, n_a)):
            host_by_action_subnet[a, :, 0] = 0.0
        # No-target (9 = Sleep): subnet=0, host=0
        if n_a > 9:
            subnet_by_action[9] = 0.0
            subnet_by_action[9, 0] = 1.0
            host_by_action_subnet[9] = 0.0
            host_by_action_subnet[9, :, 0] = 1.0
        return np.concatenate([
            action_type_mask,
            subnet_by_action.flatten(),
            host_by_action_subnet.flatten(),
        ])

    @staticmethod
    def _make_sleep_only_mask(n_a, n_s, n_h):
        """Build a hierarchical mask that permits only Sleep=(9,0,0)."""
        action_type_mask = np.zeros(n_a, dtype=np.float32)
        subnet_by_action = np.zeros((n_a, n_s), dtype=np.float32)
        host_by_action_subnet = np.zeros((n_a, n_s, n_h), dtype=np.float32)
        sleep_idx = min(9, n_a - 1)
        action_type_mask[sleep_idx] = 1.0
        subnet_by_action[sleep_idx, 0] = 1.0
        host_by_action_subnet[sleep_idx, 0, 0] = 1.0
        return np.concatenate([
            action_type_mask,
            subnet_by_action.flatten(),
            host_by_action_subnet.flatten(),
        ])

    def _build_hierarchical_mask(self, action_mask_dict):
        """Build flat hierarchical mask from per-dimension masks + structural constraints."""
        n_a, n_s, n_h = self._nvec[0], self._nvec[1], self._nvec[2]
        action_type_mask = np.array(
            action_mask_dict.get("action_type", np.ones(n_a)), dtype=np.float32)
        target_subnet = np.array(
            action_mask_dict.get("target_subnet", np.ones(n_s)), dtype=np.float32)
        target_host = np.array(
            action_mask_dict.get("target_host", np.ones(n_h)), dtype=np.float32)

        subnet_by_action = action_mask_dict.get("target_subnet_by_action")
        if subnet_by_action is None:
            subnet_by_action = np.repeat(target_subnet[None, :], n_a, axis=0)
        subnet_by_action = np.array(subnet_by_action, dtype=np.float32)

        host_by_action_subnet = action_mask_dict.get("target_host_by_action_subnet")
        if host_by_action_subnet is None:
            host_by_action_subnet = np.repeat(
                np.repeat(target_host[None, None, :], n_a, axis=0), n_s, axis=1)
        host_by_action_subnet = np.array(host_by_action_subnet, dtype=np.float32)

        # Structural constraints
        # Subnet-targeted (0): host must be 0
        host_by_action_subnet[self.SUBNET_ACTION_TYPE, :, 1:] = 0.0
        host_by_action_subnet[self.SUBNET_ACTION_TYPE, :, 0] = np.maximum(
            host_by_action_subnet[self.SUBNET_ACTION_TYPE, :, 0], 1.0)
        # Host-targeted (1-8): host must be > 0
        for a in self.HOST_ACTION_TYPES:
            if a < n_a:
                host_by_action_subnet[a, :, 0] = 0.0
        # No-target (Sleep=9): subnet=0, host=0
        if n_a > self.NO_TARGET_ACTION_TYPE:
            subnet_by_action[self.NO_TARGET_ACTION_TYPE] = 0.0
            subnet_by_action[self.NO_TARGET_ACTION_TYPE, 0] = 1.0
            host_by_action_subnet[self.NO_TARGET_ACTION_TYPE] = 0.0
            host_by_action_subnet[self.NO_TARGET_ACTION_TYPE, :, 0] = 1.0

        return np.concatenate([
            action_type_mask,
            subnet_by_action.flatten(),
            host_by_action_subnet.flatten(),
        ])

    def _update_available_actions(self, info):
        """Extract action masks from env info and store as flat hierarchical masks."""
        for idx, agent in enumerate(self.red_agents):
            if (agent in self._active_agents
                    and isinstance(info, dict)
                    and isinstance(info.get(agent), dict)
                    and 'action_mask' in info[agent]):
                self.available_actions[idx] = self._build_hierarchical_mask(
                    info[agent]['action_mask'])
            else:
                self.available_actions[idx] = self._inactive_mask.copy()

    def _update_active_masks(self):
        self.active_masks = np.array(
            [[1.0 if agent in self._active_agents else 0.0] for agent in self.red_agents],
            dtype=np.float32,
        )

    def _pack_obs(self, obs_dict):
        return np.stack(
            [
                np.asarray(obs_dict.get(agent, self._zero_obs), dtype=np.float32)
                for agent in self.red_agents
            ],
            axis=0,
        )

    def reset(self):
        kwargs = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        obs, info = self.env.reset(**kwargs)
        self._active_agents = set(obs.keys())
        self._update_available_actions(info)
        self._update_active_masks()
        return self._pack_obs(obs)

    def step(self, actions):
        actions = np.asarray(actions)
        action_dict = {}
        for idx, agent in enumerate(self.red_agents):
            if agent in self._active_agents:
                action_dict[agent] = actions[idx].astype(np.int64).tolist()

        obs, rewards, terms, truncs, info = self.env.step(action_dict)
        self._active_agents = set(obs.keys())
        self._update_available_actions(info)
        self._update_active_masks()
        obs_arr = self._pack_obs(obs)

        rewards_arr = np.array(
            [[float(rewards.get(agent, 0.0))] for agent in self.red_agents],
            dtype=np.float32,
        )
        done_all = bool(terms.get("__all__", False) or truncs.get("__all__", False))
        dones_arr = np.array(
            [done_all or (agent not in self._active_agents) for agent in self.red_agents],
            dtype=bool,
        )
        infos = []
        for agent in self.red_agents:
            if isinstance(info, dict):
                infos.append(info.get(agent, {}))
            else:
                infos.append({})
        return obs_arr, rewards_arr, dones_arr, infos

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


class ActionMaskVecEnv:
    """Simple vectorized environment that collects available_actions from sub-envs."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        env = self.envs[0]
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)

    def get_available_actions(self):
        """Return available_actions from all sub-envs: [n_envs, num_agents, mask_size]."""
        return np.stack([env.available_actions.copy() for env in self.envs])

    def get_active_masks(self):
        """Return active masks from all sub-envs: [n_envs, num_agents, 1]."""
        return np.stack([env.active_masks.copy() for env in self.envs])

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, done, info = env.step(actions[i])
            if np.all(done):
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        return np.stack(obs_list), np.stack(rew_list), np.stack(done_list), info_list

    def close(self):
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def _build_onpolicy_args(
    train_alg,
    num_iterations,
    num_rollout_threads,
    num_gpus,
    train_batch_size,
    sgd_minibatch_size,
    num_sgd_iter,
    share_policy=None,
    share_critic=None,
):
    from onpolicy.config import get_config

    parser = get_config()
    all_args = parser.parse_args([])
    all_args.algorithm_name = train_alg
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False
    all_args.use_centralized_V = train_alg == "mappo"
    # Default: per-agent policies for both MAPPO and IPPO.
    # MAPPO vs IPPO is distinguished by use_centralized_V, not share_policy.
    # share_policy=True (parameter sharing) is an orthogonal optimization for
    # homogeneous agents, available via --RL_SHARE_POLICY true.
    if share_policy is None:
        all_args.share_policy = False
    else:
        all_args.share_policy = share_policy
    num_agents = 6
    # Use RL_EPISODE_LENGTH (default 500) to match the CybORG env episode length.
    # Previously deriving this from train_batch_size/(threads*agents)=167 meant
    # updates only covered the early mission phase.
    all_args.episode_length = int(os.environ.get("RL_EPISODE_LENGTH", "500"))
    all_args.n_rollout_threads = int(max(1, num_rollout_threads))
    all_args.hidden_size = 1024
    all_args.layer_N = 2
    all_args.use_ReLU = True
    all_args.lr = 5e-5
    all_args.critic_lr = 5e-5
    all_args.entropy_coef = 0.01
    all_args.gamma = 0.99
    all_args.ppo_epoch = int(max(1, num_sgd_iter))
    # Shared mode: batch = T * N_threads * N_agents
    # Separated mode: batch = T * N_threads (per agent)
    if all_args.share_policy:
        total_samples = all_args.episode_length * all_args.n_rollout_threads * num_agents
    else:
        total_samples = all_args.episode_length * all_args.n_rollout_threads
    all_args.num_mini_batch = int(
        max(1, np.ceil(float(total_samples) / float(max(1, sgd_minibatch_size))))
    )
    all_args.n_training_threads = 1
    all_args.num_env_steps = int(
        all_args.episode_length * all_args.n_rollout_threads * max(1, int(num_iterations))
    )
    all_args.cuda = bool(num_gpus > 0 and torch and torch.cuda.is_available())
    all_args.use_wandb = False
    all_args.env_name = "CybORG"
    all_args.scenario_name = "Enterprise"
    all_args.experiment_name = f"train_red_ppo_{train_alg}"
    # Shared critic: default True for MAPPO separated mode, False otherwise.
    if share_critic is None:
        all_args.share_critic = (train_alg == "mappo" and not all_args.share_policy)
    else:
        all_args.share_critic = share_critic
    return all_args


class CybORGMAPPORunner:
    def __init__(self, all_args, envs, device):
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
        from onpolicy.utils.shared_buffer import SharedReplayBuffer
        from onpolicy.utils.separated_buffer import SeparatedReplayBuffer

        self.all_args = all_args
        self.envs = envs
        self.device = device
        self.n_rollout_threads = all_args.n_rollout_threads
        self.num_agents = 6
        self.episode_length = all_args.episode_length
        self.hidden_size = all_args.hidden_size
        self.recurrent_N = all_args.recurrent_N
        self.use_centralized_V = all_args.use_centralized_V
        self.share_policy = all_args.share_policy

        share_obs_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        if self.share_policy:
            # Shared mode (standard MAPPO): 1 policy, 1 trainer, 1 SharedReplayBuffer(num_agents=6)
            self.policy = R_MAPPOPolicy(
                all_args,
                self.envs.observation_space[0],
                share_obs_space,
                self.envs.action_space[0],
                device=device,
            )
            self.trainer = R_MAPPO(all_args, self.policy, device=device)
            self.buffer = SharedReplayBuffer(
                all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_obs_space,
                self.envs.action_space[0],
            )
        else:
            # Separated mode (IPPO): per-agent policies, trainers, SeparatedReplayBuffers
            self.policies = []
            self.trainers = []
            self.buffers = []
            for _ in range(self.num_agents):
                policy = R_MAPPOPolicy(
                    all_args,
                    self.envs.observation_space[0],
                    share_obs_space,
                    self.envs.action_space[0],
                    device=device,
                )
                trainer = R_MAPPO(all_args, policy, device=device)
                buf = SeparatedReplayBuffer(
                    all_args,
                    self.envs.observation_space[0],
                    share_obs_space,
                    self.envs.action_space[0],
                )
                self.policies.append(policy)
                self.trainers.append(trainer)
                self.buffers.append(buf)

            # Shared critic: reuse policies[0]'s critic, critic_optimizer,
            # and value_normalizer across all agents.
            self.share_critic = getattr(all_args, "share_critic", False)
            if self.share_critic and not self.share_policy:
                shared_critic = self.policies[0].critic
                shared_critic_optimizer = self.policies[0].critic_optimizer
                shared_value_normalizer = self.trainers[0].value_normalizer
                for idx in range(1, self.num_agents):
                    self.policies[idx].critic = shared_critic
                    self.policies[idx].critic_optimizer = shared_critic_optimizer
                    self.trainers[idx].value_normalizer = shared_value_normalizer

    def _get_share_obs_separated(self, obs, agent_idx):
        """Build shared observation for separated mode.

        CTDE: centralized critic sees all agents' obs concatenated (use_centralized_V=True/MAPPO),
        decentralized critic sees only own obs (use_centralized_V=False/IPPO).
        Returns shape [N, share_obs_dim].
        """
        if self.use_centralized_V:
            return obs.reshape(self.n_rollout_threads, -1)
        else:
            return obs[:, agent_idx]

    def warmup(self):
        obs = self.envs.reset()  # [N, num_agents, obs_dim]
        avail_actions = self.envs.get_available_actions()  # [N, num_agents, mask_size]
        active_masks = self.envs.get_active_masks()  # [N, num_agents, 1]

        if self.share_policy:
            # Shared mode: buffer shape [T+1, N, num_agents, ...]
            if self.use_centralized_V:
                share_obs = obs.reshape(self.n_rollout_threads, -1)  # [N, num_agents*obs_dim]
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  # [N, num_agents, share_obs_dim]
            else:
                share_obs = obs.copy()
            self.buffer.share_obs[0] = share_obs.copy()
            self.buffer.obs[0] = obs.copy()
            self.buffer.active_masks[0] = active_masks.copy()
            if self.buffer.available_actions is not None:
                self.buffer.available_actions[0] = avail_actions.copy()
        else:
            # Separated mode: per-agent buffers, shape [T+1, N, ...]
            for agent_idx in range(self.num_agents):
                agent_share_obs = self._get_share_obs_separated(obs, agent_idx)
                self.buffers[agent_idx].share_obs[0] = agent_share_obs.copy()
                self.buffers[agent_idx].obs[0] = obs[:, agent_idx].copy()
                self.buffers[agent_idx].active_masks[0] = active_masks[:, agent_idx].copy()
                if self.buffers[agent_idx].available_actions is not None:
                    self.buffers[agent_idx].available_actions[0] = avail_actions[:, agent_idx].copy()

    @torch.no_grad()
    def collect(self, step):
        if self.share_policy:
            return self._collect_shared(step)
        else:
            return self._collect_separated(step)

    @torch.no_grad()
    def _collect_shared(self, step):
        """Shared mode: single forward pass through shared policy for all agents."""
        self.trainer.prep_rollout()
        buf = self.buffer
        N = self.n_rollout_threads
        n_agents = self.num_agents

        # Concatenate across agents: [N, num_agents, dim] -> [N*num_agents, dim]
        share_obs = buf.share_obs[step].reshape(N * n_agents, -1)
        obs = buf.obs[step].reshape(N * n_agents, -1)
        rnn_states = buf.rnn_states[step].reshape(N * n_agents, *buf.rnn_states.shape[3:])
        rnn_states_critic = buf.rnn_states_critic[step].reshape(N * n_agents, *buf.rnn_states_critic.shape[3:])
        masks = buf.masks[step].reshape(N * n_agents, -1)
        avail = None
        if buf.available_actions is not None:
            avail = buf.available_actions[step].reshape(N * n_agents, -1)

        value, action, action_log_prob, rnn_s, rnn_sc = self.policy.get_actions(
            share_obs, obs, rnn_states, rnn_states_critic, masks,
            available_actions=avail,
        )

        # Split back to [N, num_agents, ...]
        values = np.array(np.split(value.detach().cpu().numpy(), N))
        actions = np.array(np.split(action.detach().cpu().numpy(), N))
        action_log_probs = np.array(np.split(action_log_prob.detach().cpu().numpy(), N))
        rnn_states_out = np.array(np.split(rnn_s.detach().cpu().numpy(), N))
        rnn_states_critic_out = np.array(np.split(rnn_sc.detach().cpu().numpy(), N))
        return values, actions, action_log_probs, rnn_states_out, rnn_states_critic_out

    @torch.no_grad()
    def _collect_separated(self, step):
        """Separated mode: per-agent forward pass."""
        value_list, action_list, action_log_prob_list = [], [], []
        rnn_state_list, rnn_state_critic_list = [], []
        for agent_idx in range(self.num_agents):
            trainer = self.trainers[agent_idx]
            policy = self.policies[agent_idx]
            buf = self.buffers[agent_idx]
            trainer.prep_rollout()

            avail = None
            if buf.available_actions is not None:
                avail = buf.available_actions[step]  # [N, mask_size]

            value, action, action_log_prob, rnn_s, rnn_sc = policy.get_actions(
                buf.share_obs[step],  # [N, share_obs_dim]
                buf.obs[step],        # [N, obs_dim]
                buf.rnn_states[step],  # [N, recurrent_N, hidden_size]
                buf.rnn_states_critic[step],
                buf.masks[step],       # [N, 1]
                available_actions=avail,
            )
            value_list.append(value.detach().cpu().numpy())
            action_list.append(action.detach().cpu().numpy())
            action_log_prob_list.append(action_log_prob.detach().cpu().numpy())
            rnn_state_list.append(rnn_s.detach().cpu().numpy())
            rnn_state_critic_list.append(rnn_sc.detach().cpu().numpy())

        # Stack to [N, num_agents, ...] for consistent return format
        values = np.stack(value_list, axis=1)
        actions = np.stack(action_list, axis=1)
        action_log_probs = np.stack(action_log_prob_list, axis=1)
        rnn_states = np.stack(rnn_state_list, axis=1)
        rnn_states_critic = np.stack(rnn_state_critic_list, axis=1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, obs, rewards, dones, values, actions, action_log_probs,
               rnn_states, rnn_states_critic, avail_actions, active_masks):
        if self.share_policy:
            self._insert_shared(obs, rewards, dones, values, actions, action_log_probs,
                                rnn_states, rnn_states_critic, avail_actions, active_masks)
        else:
            self._insert_separated(obs, rewards, dones, values, actions, action_log_probs,
                                   rnn_states, rnn_states_critic, avail_actions, active_masks)

    def _insert_shared(self, obs, rewards, dones, values, actions, action_log_probs,
                       rnn_states, rnn_states_critic, avail_actions, active_masks):
        """Shared mode: single buffer with [T+1, N, num_agents, ...] shape."""
        N = self.n_rollout_threads
        if self.use_centralized_V:
            share_obs = obs.reshape(N, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs.copy()

        # Build per-agent masks and handle RNN reset on done
        masks = np.ones((N, self.num_agents, 1), dtype=np.float32)
        rnn_out = rnn_states.copy()
        rnn_critic_out = rnn_states_critic.copy()
        for agent_idx in range(self.num_agents):
            agent_done = dones[:, agent_idx]
            masks[agent_done, agent_idx] = 0.0
            if np.any(agent_done):
                rnn_out[agent_done, agent_idx] = 0.0
                rnn_critic_out[agent_done, agent_idx] = 0.0

        self.buffer.insert(
            share_obs, obs, rnn_out, rnn_critic_out,
            actions, action_log_probs, values, rewards, masks,
            active_masks=active_masks,
            available_actions=avail_actions,
        )

    def _insert_separated(self, obs, rewards, dones, values, actions, action_log_probs,
                          rnn_states, rnn_states_critic, avail_actions, active_masks):
        """Separated mode: per-agent buffers with [T+1, N, ...] shape."""
        for agent_idx in range(self.num_agents):
            buf = self.buffers[agent_idx]
            agent_share_obs = self._get_share_obs_separated(obs, agent_idx)
            agent_done = dones[:, agent_idx]
            agent_rnn = rnn_states[:, agent_idx].copy()
            agent_rnn_c = rnn_states_critic[:, agent_idx].copy()
            if np.any(agent_done):
                agent_rnn[agent_done] = 0.0
                agent_rnn_c[agent_done] = 0.0
            agent_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            agent_masks[agent_done] = 0.0

            agent_avail = None
            if buf.available_actions is not None and avail_actions is not None:
                agent_avail = avail_actions[:, agent_idx]
            agent_active_masks = active_masks[:, agent_idx]

            buf.insert(
                agent_share_obs,
                obs[:, agent_idx],
                agent_rnn,
                agent_rnn_c,
                actions[:, agent_idx],
                action_log_probs[:, agent_idx],
                values[:, agent_idx],
                rewards[:, agent_idx],
                agent_masks,
                active_masks=agent_active_masks,
                available_actions=agent_avail,
            )

    @torch.no_grad()
    def compute(self):
        if self.share_policy:
            self._compute_shared()
        else:
            self._compute_separated()

    @torch.no_grad()
    def _compute_shared(self):
        """Shared mode: single policy computes values for all agents."""
        self.trainer.prep_rollout()
        buf = self.buffer
        N = self.n_rollout_threads
        n_agents = self.num_agents

        share_obs = buf.share_obs[-1].reshape(N * n_agents, -1)
        rnn_sc = buf.rnn_states_critic[-1].reshape(N * n_agents, *buf.rnn_states_critic.shape[3:])
        masks = buf.masks[-1].reshape(N * n_agents, -1)

        next_values = self.policy.get_values(share_obs, rnn_sc, masks)
        next_values = np.array(np.split(next_values.detach().cpu().numpy(), N))
        buf.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def _compute_separated(self):
        """Separated mode: per-agent value computation."""
        for agent_idx in range(self.num_agents):
            trainer = self.trainers[agent_idx]
            policy = self.policies[agent_idx]
            buf = self.buffers[agent_idx]
            trainer.prep_rollout()
            next_values = policy.get_values(
                buf.share_obs[-1],            # [N, share_obs_dim]
                buf.rnn_states_critic[-1],    # [N, recurrent_N, hidden_size]
                buf.masks[-1],                # [N, 1]
            )
            next_values = next_values.detach().cpu().numpy()
            buf.compute_returns(next_values, trainer.value_normalizer)

    def train_one_update(self):
        for step in range(self.episode_length):
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
            obs, rewards, dones, _ = self.envs.step(actions.astype(np.int64))
            avail_actions = self.envs.get_available_actions()
            active_masks = self.envs.get_active_masks()
            self.insert(
                obs, rewards, dones, values, actions, action_log_probs,
                rnn_states, rnn_states_critic, avail_actions, active_masks,
            )

        # Compute episode returns for logging
        if self.share_policy:
            # buffer.rewards shape: [T, N, num_agents, 1]
            episode_returns = np.sum(self.buffer.rewards[:, :, :, 0], axis=0)  # [N, num_agents]
            active_any = np.max(self.buffer.active_masks[:-1, :, :, 0], axis=0).astype(bool)
            reward_mean = float(np.mean(episode_returns[active_any])) if np.any(active_any) else 0.0
        else:
            episode_returns = []
            active_any_list = []
            for agent_idx in range(self.num_agents):
                returns = np.sum(self.buffers[agent_idx].rewards[:, :, 0], axis=0)  # [N]
                episode_returns.append(returns)
                active_any_list.append(
                    np.max(self.buffers[agent_idx].active_masks[:-1, :, 0], axis=0).astype(bool)
                )
            episode_returns = np.stack(episode_returns, axis=1)  # [N, num_agents]
            active_any = np.stack(active_any_list, axis=1)
            reward_mean = float(np.mean(episode_returns[active_any])) if np.any(active_any) else 0.0

        self.compute()

        if self.share_policy:
            train_infos = self._train_shared()
        else:
            train_infos = self._train_separated()

        return reward_mean, train_infos

    def _train_shared(self):
        """Shared mode: single trainer update."""
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        # Convert tensor values to floats
        return {k: (v.detach().cpu().item() if hasattr(v, "cpu") else float(v))
                for k, v in train_info.items()}

    def _train_separated(self):
        """Separated mode with importance sampling factor (from on-policy/runner/separated/base_runner.py)."""
        train_info_list = []
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_idx in torch.randperm(self.num_agents).tolist():
            trainer = self.trainers[agent_idx]
            policy = self.policies[agent_idx]
            buf = self.buffers[agent_idx]
            if np.sum(buf.active_masks[:-1]) <= 0.0:
                buf.after_update()
                continue

            buf.update_factor(factor)

            # Compute old log probs before training
            trainer.prep_rollout()
            with torch.no_grad():
                old_actions_logprob, _ = policy.actor.evaluate_actions(
                    buf.obs[:-1].reshape(-1, *buf.obs.shape[2:]),
                    buf.rnn_states[:-1].reshape(-1, *buf.rnn_states.shape[2:]),
                    buf.actions.reshape(-1, *buf.actions.shape[2:]),
                    buf.masks[:-1].reshape(-1, *buf.masks.shape[2:]),
                    buf.available_actions[:-1].reshape(-1, *buf.available_actions.shape[2:])
                    if buf.available_actions is not None else None,
                    buf.active_masks[:-1].reshape(-1, *buf.active_masks.shape[2:]),
                )

            # Train this agent
            trainer.prep_training()
            train_info = trainer.train(buf)
            train_info_list.append(train_info)

            # Compute new log probs after training
            trainer.prep_rollout()
            with torch.no_grad():
                new_actions_logprob, _ = policy.actor.evaluate_actions(
                    buf.obs[:-1].reshape(-1, *buf.obs.shape[2:]),
                    buf.rnn_states[:-1].reshape(-1, *buf.rnn_states.shape[2:]),
                    buf.actions.reshape(-1, *buf.actions.shape[2:]),
                    buf.masks[:-1].reshape(-1, *buf.masks.shape[2:]),
                    buf.available_actions[:-1].reshape(-1, *buf.available_actions.shape[2:])
                    if buf.available_actions is not None else None,
                    buf.active_masks[:-1].reshape(-1, *buf.active_masks.shape[2:]),
                )

            # Update importance sampling factor
            logprob_diff = (new_actions_logprob - old_actions_logprob).detach().cpu().numpy()
            # Sum across action dimensions if multi-dimensional
            if logprob_diff.ndim > 1:
                logprob_diff = logprob_diff.sum(axis=-1, keepdims=True)
            factor_update = np.exp(logprob_diff).reshape(self.episode_length, self.n_rollout_threads, 1)
            factor = factor * factor_update

            buf.after_update()

        # Average training info across agents
        train_infos = {}
        if train_info_list:
            for key in train_info_list[0]:
                train_infos[key] = float(np.mean([
                    (v.detach().cpu().item() if hasattr(v, "cpu") else float(v))
                    for info in train_info_list for v in [info[key]]
                ]))
        return train_infos

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        share_critic = getattr(self, "share_critic", False)
        if self.share_policy:
            torch.save(self.policy.actor.state_dict(), os.path.join(save_dir, "actor.pt"))
            torch.save(self.policy.critic.state_dict(), os.path.join(save_dir, "critic.pt"))
        else:
            for agent_idx, policy in enumerate(self.policies):
                torch.save(policy.actor.state_dict(), os.path.join(save_dir, f"actor_agent_{agent_idx}.pt"))
                if not share_critic:
                    torch.save(policy.critic.state_dict(), os.path.join(save_dir, f"critic_agent_{agent_idx}.pt"))
            if share_critic:
                torch.save(self.policies[0].critic.state_dict(), os.path.join(save_dir, "shared_critic.pt"))
        # Write metadata
        meta = {"share_policy": self.share_policy, "share_critic": share_critic}
        with open(os.path.join(save_dir, "runner_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--RL_TRAIN_ALG",
        type=str,
        default="ippo",
        choices=["ippo", "mappo"],
        help="Training algorithm (ippo or mappo)",
    )
    parser.add_argument("--RL_SAVE_DIR", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--RL_NUM_GPUS", type=float, default=1.0)
    parser.add_argument("--RL_NUM_ROLLOUT_WORKERS", type=int, default=2)
    parser.add_argument("--RL_NUM_ITERATIONS", type=int, default=20)
    parser.add_argument("--RL_CHECKPOINT_INTERVAL", type=int, default=10)
    parser.add_argument(
        "--RL_SHARE_POLICY",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Share policy across agents (auto: true for mappo, false for ippo)",
    )
    parser.add_argument(
        "--RL_SHARE_CRITIC",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Share critic across agents in separated mode (auto: true for mappo, false for ippo)",
    )
    parser.add_argument("--WANDB_PROJECT", type=str, default="cc4", help="W&B project name")
    parser.add_argument("--WANDB_RUN_NAME", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--BLUE_OPPONENT",
        type=str,
        default=os.environ.get("BLUE_OPPONENT", "sleep"),
        choices=["sleep", "marl"],
        help="Blue opponent for red training. Use 'marl' after training a blue checkpoint.",
    )
    args = parser.parse_args()

    # Update environment variables for internal use
    os.environ["RL_TRAIN_ALG"] = args.RL_TRAIN_ALG
    os.environ["BLUE_OPPONENT"] = args.BLUE_OPPONENT

    train_alg = args.RL_TRAIN_ALG.lower()
    if args.RL_SAVE_DIR:
        save_dir = args.RL_SAVE_DIR
    else:
        save_dir = str(_REPO_ROOT / "red_checkpoints" / "ppo" / train_alg)
        
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.environ.get("RL_TRAIN_LOG_FILE", "training.log")
    log_path = os.path.join(save_dir, log_filename)
    logger = setup_training_logger(log_path)
    logger.info("Training log file: %s", log_path)
    warnings.filterwarnings(
        "ignore",
        message=r".*_get_slice_indices.*deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"overflow encountered in reduce",
        category=RuntimeWarning,
        module=r"numpy\.core\._methods",
    )

    try:
        import wandb
    except Exception as e:
        wandb = None
        logger.warning("WandB disabled: %s", e)
    
    if os.environ.get("RL_USE_LEGACY_ONPOLICY") == "0":
        logger.warning(
            "RL_USE_LEGACY_ONPOLICY=0 is ignored: train_red_ppo only supports the on-policy IPPO/MAPPO runner."
        )
    need_ray = False
    if need_ray and not ray.is_initialized():
        ray_num_cpus = _resolve_ray_num_cpus()
        logger.info(
            "Initializing Ray with num_cpus=%d (RL_RAY_NUM_CPUS=%s, SLURM_CPUS_PER_TASK=%s)",
            ray_num_cpus,
            os.environ.get("RL_RAY_NUM_CPUS"),
            os.environ.get("SLURM_CPUS_PER_TASK"),
        )
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
        )

    logger.info("Setting up algorithm config...")
    num_gpus = args.RL_NUM_GPUS
    num_rollout_workers = args.RL_NUM_ROLLOUT_WORKERS
    num_iterations = args.RL_NUM_ITERATIONS
    train_batch_size = int(os.environ.get("RL_TRAIN_BATCH_SIZE", "4000"))
    sgd_minibatch_size = int(os.environ.get("RL_SGD_MINIBATCH_SIZE", "128"))
    num_sgd_iter = int(os.environ.get("RL_NUM_SGD_ITER", "10"))
    midterm_episode_interval = args.RL_CHECKPOINT_INTERVAL
    save_best_checkpoint = os.environ.get("RL_SAVE_BEST_CHECKPOINT", "1") != "0"
    best_metric_name = os.environ.get("RL_BEST_METRIC", "reward_mean")
    best_metric_mode = os.environ.get("RL_BEST_METRIC_MODE", "max").strip().lower()
    if best_metric_mode not in {"max", "min"}:
        logger.warning(
            "Invalid RL_BEST_METRIC_MODE=%s, fallback to 'max'",
            best_metric_mode,
        )
        best_metric_mode = "max"
    best_min_iteration = int(os.environ.get("RL_BEST_MIN_ITERATION", "1"))
    best_meta_filename = os.environ.get("RL_BEST_META_FILE", "best_checkpoint.json")
    best_meta_path = os.path.join(save_dir, best_meta_filename)
    best_checkpoint_dir = os.path.join(save_dir, "best")
    latest_checkpoint_dir = os.path.join(save_dir, "latest")
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    os.makedirs(latest_checkpoint_dir, exist_ok=True)
    cuda_available = bool(torch and torch.cuda.is_available())
    logger.info(
        (
            "Training algorithm: %s, resources: num_gpus=%s, num_rollout_workers=%s, num_iterations=%s, "
            "train_batch_size=%s, sgd_minibatch_size=%s, num_sgd_iter=%s, "
            "midterm_episode_interval=%s, save_best=%s, best_metric=%s, best_mode=%s, "
            "best_min_iteration=%s, best_checkpoint_dir=%s, latest_checkpoint_dir=%s, cuda_available=%s"
        ),
        train_alg,
        num_gpus,
        num_rollout_workers,
        num_iterations,
        train_batch_size,
        sgd_minibatch_size,
        num_sgd_iter,
        midterm_episode_interval,
        save_best_checkpoint,
        best_metric_name,
        best_metric_mode,
        best_min_iteration,
        best_checkpoint_dir,
        latest_checkpoint_dir,
        cuda_available,
    )
    
    if wandb is not None:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = args.WANDB_RUN_NAME if args.WANDB_RUN_NAME else f"red_ppo_{train_alg}_{timestamp}"
            wandb.init(
                project=args.WANDB_PROJECT,
                sync_tensorboard=True,
                name=run_name,
                config={
                    "train_alg": train_alg,
                    "num_gpus": num_gpus,
                    "num_rollout_workers": num_rollout_workers,
                    "train_batch_size": train_batch_size,
                    "lr": 5e-5,
                    "gamma": 0.99,
                    "entropy_coeff": 0.01,
                    "sgd_minibatch_size": sgd_minibatch_size,
                    "num_sgd_iter": num_sgd_iter,
                    "num_iterations": num_iterations,
                    "midterm_episode_interval": midterm_episode_interval,
                    "save_best_checkpoint": save_best_checkpoint,
                    "best_metric_name": best_metric_name,
                    "best_metric_mode": best_metric_mode,
                    "best_min_iteration": best_min_iteration,
                    "best_checkpoint_dir": best_checkpoint_dir,
                    "latest_checkpoint_dir": latest_checkpoint_dir,
                },
            )
        except Exception as e:
            logger.warning("WandB init failed, disabling WandB: %s", e)
            wandb = None
    
    use_onpolicy_runner = train_alg in {"mappo", "ippo"}
    if use_onpolicy_runner:
        logger.info("Setting up %s config and vectorized environments...", train_alg.upper())

        share_policy_override = None
        if args.RL_SHARE_POLICY is not None:
            share_policy_override = args.RL_SHARE_POLICY.lower() == "true"

        share_critic_override = None
        if args.RL_SHARE_CRITIC is not None:
            share_critic_override = args.RL_SHARE_CRITIC.lower() == "true"

        onpolicy_args = _build_onpolicy_args(
            train_alg,
            num_iterations,
            num_rollout_workers,
            num_gpus,
            train_batch_size,
            sgd_minibatch_size,
            num_sgd_iter,
            share_policy=share_policy_override,
            share_critic=share_critic_override,
        )
        logger.info(
            "share_policy=%s (override=%s), share_critic=%s (override=%s)",
            onpolicy_args.share_policy, args.RL_SHARE_POLICY,
            onpolicy_args.share_critic, args.RL_SHARE_CRITIC,
        )
        if onpolicy_args.cuda and torch and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.set_num_threads(onpolicy_args.n_training_threads)
        else:
            device = torch.device("cpu")
            torch.set_num_threads(onpolicy_args.n_training_threads)

        def get_env_fn(rank):
            def init_env():
                return CybORGRedMAPPOEnv(seed=onpolicy_args.seed + rank * 1000)
            return init_env

        envs = ActionMaskVecEnv(
            [get_env_fn(i) for i in range(onpolicy_args.n_rollout_threads)]
        )

        runner = CybORGMAPPORunner(onpolicy_args, envs, device)
        runner.warmup()
        logger.info("%s runner initialized.", train_alg.upper())

        try:
            run_start_ts = time.time()
            best_metric_value = None
            best_metric_iteration = None
            best_checkpoint_path = None
            latest_checkpoint_path = None

            for i in range(num_iterations):
                iter_start_ts = time.time()
                logger.info("Iteration %d/%d started...", i + 1, num_iterations)
                reward_mean, train_infos = runner.train_one_update()
                iter_wall_s = time.time() - iter_start_ts
                run_elapsed_s = time.time() - run_start_ts
                avg_iter_s = run_elapsed_s / float(i + 1)
                eta_s = avg_iter_s * float(num_iterations - (i + 1))

                opt_metrics = {
                    "policy_loss": _to_finite_float(train_infos.get("policy_loss")),
                    "value_loss": _to_finite_float(train_infos.get("value_loss")),
                    "entropy": _to_finite_float(train_infos.get("dist_entropy")),
                    "ratio": _to_finite_float(train_infos.get("ratio")),
                }
                opt_metrics = {k: v for k, v in opt_metrics.items() if v is not None}

                logger.info(
                    "Iteration %d/%d done | reward_mean=%s | iter_wall_s=%.2f | elapsed=%s | eta=%s | opt=%s",
                    i + 1,
                    num_iterations,
                    reward_mean,
                    iter_wall_s,
                    _format_seconds(run_elapsed_s),
                    _format_seconds(eta_s),
                    opt_metrics,
                )

                current_best_metric = _resolve_best_metric_value(
                    best_metric_name,
                    reward_mean,
                    opt_metrics,
                    {"episode_reward_mean": reward_mean},
                )

                if save_best_checkpoint and (i + 1) >= best_min_iteration:
                    if _is_better_metric(current_best_metric, best_metric_value, best_metric_mode):
                        best_metric_value = current_best_metric
                        best_metric_iteration = i + 1
                        logger.info(
                            (
                                "New best checkpoint at iteration %d/%d: "
                                "%s=%s (mode=%s). Saving..."
                            ),
                            i + 1,
                            num_iterations,
                            best_metric_name,
                            best_metric_value,
                            best_metric_mode,
                        )
                        best_checkpoint_path = runner.save(best_checkpoint_dir)
                        best_payload = {
                            "best_metric_name": best_metric_name,
                            "best_metric_mode": best_metric_mode,
                            "best_metric_value": best_metric_value,
                            "best_metric_iteration": best_metric_iteration,
                            "best_checkpoint_path": best_checkpoint_path,
                        }
                        with open(best_meta_path, "w", encoding="utf-8") as f:
                            json.dump(best_payload, f, indent=2)
                        logger.info(
                            "Best checkpoint updated: %s (metadata: %s)",
                            best_checkpoint_path,
                            best_meta_path,
                        )

                if wandb is not None:
                    wandb_payload = {
                        "train/iteration": i,
                        "train/progress_pct": 100.0 * float(i + 1) / float(max(num_iterations, 1)),
                        "train/reward_mean": _to_finite_float(reward_mean),
                        "train/iter_wall_s": _to_finite_float(iter_wall_s),
                        "train/elapsed_s": _to_finite_float(run_elapsed_s),
                        "train/eta_s": _to_finite_float(eta_s),
                        "train/best_metric_current": _to_finite_float(current_best_metric),
                        "train/best_metric_so_far": _to_finite_float(best_metric_value),
                    }
                    for key, value in opt_metrics.items():
                        wandb_payload[f"train/opt/{key}"] = value
                    wandb_payload = {k: v for k, v in wandb_payload.items() if v is not None}
                    wandb.log(wandb_payload, step=i)

                if midterm_episode_interval > 0 and (i + 1) % midterm_episode_interval == 0:
                    logger.info(
                        "Iteration %d/%d midterm checkpoint saving to latest...",
                        i + 1,
                        num_iterations,
                    )
                    latest_checkpoint_path = runner.save(latest_checkpoint_dir)
                    logger.info(
                        "Iteration %d/%d midterm checkpoint saved to %s",
                        i + 1,
                        num_iterations,
                        latest_checkpoint_path,
                    )

            if save_best_checkpoint:
                logger.info(
                    "Best checkpoint summary: metric=%s mode=%s value=%s iteration=%s checkpoint=%s",
                    best_metric_name,
                    best_metric_mode,
                    best_metric_value,
                    best_metric_iteration,
                    best_checkpoint_path,
                )
                if wandb is not None and wandb.run is not None:
                    wandb.run.summary["best/metric_name"] = best_metric_name
                    wandb.run.summary["best/metric_mode"] = best_metric_mode
                    if best_metric_value is not None:
                        wandb.run.summary["best/metric_value"] = best_metric_value
                    if best_metric_iteration is not None:
                        wandb.run.summary["best/iteration"] = best_metric_iteration
                    if best_checkpoint_path is not None:
                        wandb.run.summary["best/checkpoint_path"] = str(best_checkpoint_path)

            logger.info("Final checkpoint saving to latest...")
            latest_checkpoint_path = runner.save(latest_checkpoint_dir)
            logger.info("Training complete. Final/latest model saved to %s", latest_checkpoint_path)
            if wandb is not None and wandb.run is not None:
                wandb.run.summary["final/checkpoint_path"] = str(latest_checkpoint_path)
        finally:
            envs.close()
            if wandb is not None:
                wandb.finish()
            if ray.is_initialized():
                ray.shutdown()
        sys.exit(0)

    logger.info("Creating dummy environment for space probing...")
    dummy_env = env_creator({})
    dummy_env.reset()
    logger.info("Dummy environment ready.")
    
    # We need spaces for a Red agent.
    obs_space = dummy_env.observation_space('red_agent_0')
    act_space = dummy_env.action_space('red_agent_0')
    
    if train_alg == "ippo":
        # IPPO: Create per-agent policies (each agent has its own policy)
        red_agent_ids = [f"red_agent_{i}" for i in range(6)]
        policies = {
            f"{agent_id}_policy": PolicySpec(
                policy_class=PPOTorchPolicy,
                observation_space=obs_space,
                action_space=act_space,
                config={},
            )
            for agent_id in red_agent_ids
        }
        policies_to_train = list(policies.keys())
    else:
        # PPO: Single shared policy for all agents
        policies = {
            "red_policy": PolicySpec(
                policy_class=PPOTorchPolicy,
                observation_space=obs_space,
                action_space=act_space,
                config={},
            )
        }
        policies_to_train = ["red_policy"]

    config = PPOConfig()
    config.sgd_minibatch_size = sgd_minibatch_size
    config.num_sgd_iter = num_sgd_iter
    
    config = (
        config
        .resources(
            num_gpus=num_gpus,
            num_gpus_per_worker=0.0,
        )
        .environment("CybORG_Red_PPO")
        .framework("torch")
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            batch_mode="complete_episodes",
            rollout_fragment_length=100
        ) 
        .training(
            model={
                "custom_model": "action_mask_model",
                "custom_action_dist": "red_conditional_dist",
                "fcnet_hiddens": [1024, 1024],
                "fcnet_activation": "relu",
            },
            train_batch_size=train_batch_size,
            lr=5e-5,
            gamma=0.99,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
    )

    logger.info("PPO config assembled. Building algorithm...")
    algo = config.build()
    logger.info("Algorithm build complete.")

    logger.info("Starting red PPO training with blue_opponent=%s", args.BLUE_OPPONENT)
    logger.info("Training loop started: total_iterations=%d", num_iterations)

    try:
        run_start_ts = time.time()
        best_metric_value = None
        best_metric_iteration = None
        best_checkpoint_path = None
        latest_checkpoint_path = None
        last_midterm_episode_bucket = 0

        # Train for more iterations to overcome the harder opponent
        for i in range(num_iterations):
            iter_start_ts = time.time()
            logger.info("Iteration %d/%d started...", i + 1, num_iterations)
            result = algo.train()
            reward_mean = None
            if "env_runners" in result and isinstance(result["env_runners"], dict):
                reward_mean = result["env_runners"].get("episode_reward_mean")
            if reward_mean is None:
                reward_mean = result.get("episode_reward_mean")
            iter_time_s = result.get("time_this_iter_s")
            if iter_time_s is None and isinstance(result.get("timers"), dict):
                iter_time_ms = result["timers"].get("training_iteration_time_ms")
                if iter_time_ms is not None:
                    iter_time_s = float(iter_time_ms) / 1000.0
            iter_wall_s = time.time() - iter_start_ts
            run_elapsed_s = time.time() - run_start_ts
            avg_iter_s = run_elapsed_s / float(i + 1)
            eta_s = avg_iter_s * float(num_iterations - (i + 1))

            opt_metrics = extract_optimization_metrics(result)
            logger.info(
                "Iteration %d/%d done | reward_mean=%s | iter_time_s=%s | iter_wall_s=%.2f | elapsed=%s | eta=%s | env_steps=%s | agent_steps=%s | opt=%s",
                i + 1,
                num_iterations,
                reward_mean,
                iter_time_s,
                iter_wall_s,
                _format_seconds(run_elapsed_s),
                _format_seconds(eta_s),
                result.get("num_env_steps_sampled"),
                result.get("num_agent_steps_sampled"),
                opt_metrics,
            )

            current_best_metric = _resolve_best_metric_value(
                best_metric_name,
                reward_mean,
                opt_metrics,
                result,
            )

            if save_best_checkpoint and (i + 1) >= best_min_iteration:
                if _is_better_metric(current_best_metric, best_metric_value, best_metric_mode):
                    best_metric_value = current_best_metric
                    best_metric_iteration = i + 1
                    logger.info(
                        (
                            "New best checkpoint at iteration %d/%d: "
                            "%s=%s (mode=%s). Saving..."
                        ),
                        i + 1,
                        num_iterations,
                        best_metric_name,
                        best_metric_value,
                        best_metric_mode,
                    )
                    best_save_result = algo.save(best_checkpoint_dir)
                    best_checkpoint_path = _checkpoint_result_path(best_save_result)
                    if best_checkpoint_path is None:
                        logger.warning(
                            "Unable to resolve best checkpoint path from save result type=%s",
                            type(best_save_result).__name__,
                        )
                    best_payload = {
                        "best_metric_name": best_metric_name,
                        "best_metric_mode": best_metric_mode,
                        "best_metric_value": best_metric_value,
                        "best_metric_iteration": best_metric_iteration,
                        "best_checkpoint_path": best_checkpoint_path,
                    }
                    with open(best_meta_path, "w", encoding="utf-8") as f:
                        json.dump(best_payload, f, indent=2)
                    logger.info(
                        "Best checkpoint updated: %s (metadata: %s)",
                        best_checkpoint_path,
                        best_meta_path,
                    )

            if wandb is not None:
                wandb_payload = {
                    "train/iteration": i,
                    "train/progress_pct": 100.0 * float(i + 1) / float(max(num_iterations, 1)),
                    "train/reward_mean": _to_finite_float(reward_mean),
                    "train/iter_time_s": _to_finite_float(iter_time_s),
                    "train/iter_wall_s": _to_finite_float(iter_wall_s),
                    "train/elapsed_s": _to_finite_float(run_elapsed_s),
                    "train/eta_s": _to_finite_float(eta_s),
                    "train/env_steps_sampled": _to_finite_float(result.get("num_env_steps_sampled")),
                    "train/agent_steps_sampled": _to_finite_float(result.get("num_agent_steps_sampled")),
                    "train/best_metric_current": _to_finite_float(current_best_metric),
                    "train/best_metric_so_far": _to_finite_float(best_metric_value),
                }
                for key, value in opt_metrics.items():
                    wandb_payload[f"train/opt/{key}"] = value
                wandb_payload = {k: v for k, v in wandb_payload.items() if v is not None}
                wandb.log(wandb_payload, step=i)

            episodes_total = result.get("episodes_total")
            episodes_total_int = None
            if episodes_total is not None:
                try:
                    episodes_total_int = int(episodes_total)
                except (TypeError, ValueError):
                    episodes_total_int = None

            should_save_midterm = False
            if midterm_episode_interval > 0:
                if episodes_total_int is not None:
                    current_bucket = episodes_total_int // midterm_episode_interval
                    if current_bucket > last_midterm_episode_bucket:
                        should_save_midterm = True
                        last_midterm_episode_bucket = current_bucket
                elif (i + 1) % midterm_episode_interval == 0:
                    should_save_midterm = True

            if should_save_midterm:
                logger.info(
                    "Iteration %d/%d midterm checkpoint saving to latest... episodes_total=%s",
                    i + 1,
                    num_iterations,
                    episodes_total_int,
                )
                latest_save_result = algo.save(latest_checkpoint_dir)
                latest_checkpoint_path = _checkpoint_result_path(latest_save_result)
                if latest_checkpoint_path is None:
                    logger.warning(
                        "Unable to resolve latest checkpoint path from save result type=%s",
                        type(latest_save_result).__name__,
                    )
                logger.info(
                    "Iteration %d/%d midterm checkpoint saved to %s",
                    i + 1,
                    num_iterations,
                    latest_checkpoint_path,
                )

        if save_best_checkpoint:
            logger.info(
                "Best checkpoint summary: metric=%s mode=%s value=%s iteration=%s checkpoint=%s",
                best_metric_name,
                best_metric_mode,
                best_metric_value,
                best_metric_iteration,
                best_checkpoint_path,
            )
            if wandb is not None and wandb.run is not None:
                wandb.run.summary["best/metric_name"] = best_metric_name
                wandb.run.summary["best/metric_mode"] = best_metric_mode
                if best_metric_value is not None:
                    wandb.run.summary["best/metric_value"] = best_metric_value
                if best_metric_iteration is not None:
                    wandb.run.summary["best/iteration"] = best_metric_iteration
                if best_checkpoint_path is not None:
                    wandb.run.summary["best/checkpoint_path"] = str(best_checkpoint_path)

        logger.info("Final checkpoint saving to latest...")
        final_save_result = algo.save(latest_checkpoint_dir)
        latest_checkpoint_path = _checkpoint_result_path(final_save_result)
        if latest_checkpoint_path is None:
            logger.warning(
                "Unable to resolve final checkpoint path from save result type=%s",
                type(final_save_result).__name__,
            )
        logger.info("Training complete. Final/latest model saved to %s", latest_checkpoint_path)
        if wandb is not None and wandb.run is not None:
            if latest_checkpoint_path is not None:
                wandb.run.summary["final/checkpoint_path"] = str(latest_checkpoint_path)
    finally:
        if wandb is not None:
            wandb.finish()
        ray.shutdown()
