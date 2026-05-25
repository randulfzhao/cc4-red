import inspect
import time
import os
import sys
from pathlib import Path

from statistics import mean, stdev
from typing import Any
from rich import print
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from EnterpriseMAE_CC4 import EnterpriseMAE

from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPO, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
try:
    from ray.rllib.utils import check_env
except ImportError:
    check_env = None

from action_mask_model_CC4 import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

from typing import Dict, Tuple

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

ModelCatalog.register_custom_model(
    "my_model", TorchActionMaskModel
)


def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    cyborg = EnterpriseMAE(cyborg)
    return cyborg


NUM_AGENTS = 5
BLUE_NUM_ROLLOUT_WORKERS = int(os.environ.get("BLUE_NUM_ROLLOUT_WORKERS", "30"))
BLUE_TRAIN_BATCH_SIZE = int(os.environ.get("BLUE_TRAIN_BATCH_SIZE", "1000000"))
BLUE_SGD_MINIBATCH_SIZE = int(os.environ.get("BLUE_SGD_MINIBATCH_SIZE", "32768"))
BLUE_NUM_ITERATIONS = int(os.environ.get("BLUE_NUM_ITERATIONS", "200"))
BLUE_MODEL_DIR = os.environ.get("BLUE_MODEL_DIR", "models/train_marl")

# mapping to the policy directory name
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]


def build_algo_config(env):
    # Note: RLlib expects all blue policies to expose compatible observation
    # structures, even though the action masks differ by agent.
    return (
        PPOConfig()
        .framework("torch")
        .debugging(logger_config={"logdir": "logs/train_marl", "type": "ray.tune.logger.TBXLogger"})
        .environment(env="CC4")
        .experimental(
            _disable_preprocessor_api=True,
        )
        .rollouts(
            batch_mode="complete_episodes",
            num_rollout_workers=BLUE_NUM_ROLLOUT_WORKERS,
        )
        .training(
            model={"custom_model": "my_model"},
            sgd_minibatch_size=BLUE_SGD_MINIBATCH_SIZE,
            train_batch_size=BLUE_TRAIN_BATCH_SIZE,
        )
        .multi_agent(
            policies={
                ray_agent: PolicySpec(
                    policy_class=PPOTorchPolicy,
                    observation_space=env.observation_space(cyborg_agent),
                    action_space=env.action_space(cyborg_agent),
                    config={"entropy_coeff": 0.001},
                )
                for cyborg_agent, ray_agent in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper,
        )
    )


def main():
    register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
    env = env_creator_CC4({})
    algo_config = build_algo_config(env)

    if check_env is not None:
        check_env(env)

    algo = algo_config.build()
    model_dir = BLUE_MODEL_DIR
    model_dir_crt = model_dir

    for i in range(BLUE_NUM_ITERATIONS):
        iteration = i
        train_info = algo.train()
        print("\nIteration:", i, train_info)
        model_dir_crt = os.path.join(model_dir, "iter_" + str(iteration))
        print("\nSaving model in:", model_dir_crt)
        algo.save(model_dir_crt)

    algo.save(model_dir_crt)


if __name__ == "__main__":
    main()
