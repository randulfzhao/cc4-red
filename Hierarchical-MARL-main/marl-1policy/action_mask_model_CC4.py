from gymnasium.spaces import Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()

'''
Taken from Ray API Examples
Model that handles simple discrete action masking.
'''


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs, 
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )            
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        self._last_action_mask = None
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        if isinstance(action_mask, dict):
            self._last_action_mask = {
                "action_type": action_mask["action_type"],
                "target_subnet": action_mask["target_subnet"],
                "target_host": action_mask["target_host"],
            }
            if "target_subnet_by_action" in action_mask:
                self._last_action_mask["target_subnet_by_action"] = action_mask["target_subnet_by_action"]
            if "target_host_by_action_subnet" in action_mask:
                self._last_action_mask["target_host_by_action_subnet"] = action_mask["target_host_by_action_subnet"]

            mask_action = action_mask["action_type"]
            mask_subnet = action_mask["target_subnet"]
            mask_host = action_mask["target_host"]

            split_sizes = [mask_action.shape[1], mask_subnet.shape[1], mask_host.shape[1]]
            logits_action, logits_subnet, logits_host = torch.split(logits, split_sizes, dim=1)

            inf_mask_action = torch.clamp(torch.log(mask_action), min=FLOAT_MIN)
            inf_mask_subnet = torch.clamp(torch.log(mask_subnet), min=FLOAT_MIN)
            inf_mask_host = torch.clamp(torch.log(mask_host), min=FLOAT_MIN)

            masked_logits = torch.cat(
                [
                    logits_action + inf_mask_action,
                    logits_subnet + inf_mask_subnet,
                    logits_host + inf_mask_host,
                ],
                dim=1,
            )
        else:
            self._last_action_mask = None
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        #print("computing  value function")
        return self.internal_model.value_function()