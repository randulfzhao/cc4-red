import os
from CybORG.Agents import BaseAgent
from gym import Space
import subprocess
import numpy as np
import torch

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.models import ModelCatalog
from action_mask_model_CC4 import TorchActionMaskModel

ModelCatalog.register_custom_model(
    "my_model", TorchActionMaskModel
)

from serializable_policy import SerializablePolicy

class MARLAgent(BaseAgent):
    def __init__(self, name: str = None):
        super().__init__(name)

        default_root = os.path.join(os.path.dirname(__file__), "models", "train_marl")
        checkpoint_root = os.environ.get("BLUE_CHECKPOINT_ROOT", default_root)
        checkpoint_iter = os.environ.get("BLUE_CHECKPOINT_ITER", "iter_199")
        pkl_cp_dir = os.environ.get(
            "BLUE_POLICY_DIR",
            os.path.join(checkpoint_root, checkpoint_iter, "policies", self.name),
        )
        if not os.path.exists(pkl_cp_dir):
            raise FileNotFoundError(
                "Blue policy checkpoint not found at "
                f"{pkl_cp_dir}. Train blue first or set BLUE_OPPONENT=sleep "
                "for red baseline training."
            )

        print("\nLoading Serializable blue agent model from ", pkl_cp_dir)
        self.policy = SerializablePolicy.from_checkpoint(pkl_cp_dir)


    def get_action(self, observation: dict, action_space: Space):

        # Use the restored policy for serving actions.
        action, state_out, extra = self.policy.compute_single_action(obs=observation)
    
        return action
