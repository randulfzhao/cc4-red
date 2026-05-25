from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False
        self.action_type = action_space.__class__.__name__

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.mujoco_box = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_dims = [int(d) for d in action_dims]
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])

    def _has_hierarchical_mask(self, available_actions):
        """Check if available_actions is a hierarchical mask for 3-level MultiDiscrete."""
        if available_actions is None or not self.multi_discrete:
            return False
        if len(self.action_dims) != 3:
            return False
        n_a, n_s, n_h = self.action_dims
        expected = n_a + n_a * n_s + n_a * n_s * n_h
        return available_actions.shape[-1] == expected

    def _decode_hierarchical_mask(self, available_actions):
        """Decode flat available_actions into (action_mask, subnet_by_action, host_by_action_subnet)."""
        n_a, n_s, n_h = self.action_dims
        action_mask = available_actions[:, :n_a]
        subnet_by_action = available_actions[:, n_a:n_a + n_a * n_s].reshape(-1, n_a, n_s)
        host_by_action_subnet = available_actions[:, n_a + n_a * n_s:].reshape(-1, n_a, n_s, n_h)
        return action_mask, subnet_by_action, host_by_action_subnet

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            if self._has_hierarchical_mask(available_actions):
                actions, action_log_probs = self._forward_autoregressive(
                    x, available_actions, deterministic)
            else:
                actions = []
                action_log_probs = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)

                actions = torch.cat(actions, -1)
                action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.mujoco_box:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def _forward_autoregressive(self, x, available_actions, deterministic):
        """Autoregressive forward for 3-level MultiDiscrete with hierarchical masking."""
        action_mask, subnet_by_action, host_by_action_subnet = \
            self._decode_hierarchical_mask(available_actions)
        batch_idx = torch.arange(x.shape[0], device=x.device)

        # 1. Sample action type with mask
        action_dist = self.action_outs[0](x, action_mask)
        action_type = action_dist.mode() if deterministic else action_dist.sample()
        action_type_logp = action_dist.log_probs(action_type)

        # 2. Get conditional subnet mask and sample
        at = action_type.long().squeeze(-1).clamp(0, self.action_dims[0] - 1)
        subnet_mask = subnet_by_action[batch_idx, at]
        subnet_dist = self.action_outs[1](x, subnet_mask)
        subnet = subnet_dist.mode() if deterministic else subnet_dist.sample()
        subnet_logp = subnet_dist.log_probs(subnet)

        # 3. Get conditional host mask and sample
        sn = subnet.long().squeeze(-1).clamp(0, self.action_dims[1] - 1)
        host_mask = host_by_action_subnet[batch_idx, at, sn]
        host_dist = self.action_outs[2](x, host_mask)
        host = host_dist.mode() if deterministic else host_dist.sample()
        host_logp = host_dist.log_probs(host)

        actions = torch.cat([action_type, subnet, host], -1)
        action_log_probs = torch.sum(
            torch.cat([action_type_logp, subnet_logp, host_logp], -1),
            -1,
            keepdim=True)
        return actions, action_log_probs

    def _evaluate_autoregressive(self, x, action, available_actions, active_masks):
        """Autoregressive evaluate_actions for 3-level MultiDiscrete with hierarchical masking."""
        action_mask, subnet_by_action, host_by_action_subnet = \
            self._decode_hierarchical_mask(available_actions)
        # action is [n_sub_actions, batch] after transpose in caller
        batch_idx = torch.arange(x.shape[0], device=x.device)

        # Action type
        action_dist = self.action_outs[0](x, action_mask)
        action_type_logp = action_dist.log_probs(action[0])

        # Conditional subnet mask
        at = action[0].long().squeeze(-1).clamp(0, self.action_dims[0] - 1)
        subnet_mask = subnet_by_action[batch_idx, at]
        subnet_dist = self.action_outs[1](x, subnet_mask)
        subnet_logp = subnet_dist.log_probs(action[1])

        # Conditional host mask
        sn = action[1].long().squeeze(-1).clamp(0, self.action_dims[1] - 1)
        host_mask = host_by_action_subnet[batch_idx, at, sn]
        host_dist = self.action_outs[2](x, host_mask)
        host_logp = host_dist.log_probs(action[2])

        action_log_probs = torch.sum(
            torch.cat([action_type_logp, subnet_logp, host_logp], -1),
            -1,
            keepdim=True)

        if active_masks is not None:
            am = active_masks.squeeze(-1)
            dist_entropy = (
                (action_dist.entropy() * am).sum() / am.sum()
                + (subnet_dist.entropy() * am).sum() / am.sum()
                + (host_dist.entropy() * am).sum() / am.sum()
            ) / 3.0
        else:
            dist_entropy = (
                action_dist.entropy().mean()
                + subnet_dist.entropy().mean()
                + host_dist.entropy().mean()
            ) / 3.0

        return action_log_probs, dist_entropy

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            if self.multi_discrete and self._has_hierarchical_mask(available_actions):
                action_mask = available_actions[:, :self.action_dims[0]]
                action_probs = []
                action_probs.append(self.action_outs[0](x, action_mask).probs)
                for i in range(1, len(self.action_outs)):
                    action_probs.append(self.action_outs[i](x).probs)
                action_probs = torch.cat(action_probs, -1)
            else:
                action_probs = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    action_prob = action_logit.probs
                    action_probs.append(action_prob)
                action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum())
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98 #! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            if self._has_hierarchical_mask(available_actions):
                action_log_probs, dist_entropy = self._evaluate_autoregressive(
                    x, action, available_actions, active_masks)
            else:
                action_log_probs = []
                dist_entropy = []
                for action_out, act in zip(self.action_outs, action):
                    action_logit = action_out(x)
                    action_log_probs.append(action_logit.log_probs(act))
                    if active_masks is not None:
                        dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())

                action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
                dist_entropy = sum(dist_entropy)/len(dist_entropy)

        elif self.mujoco_box:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            if self._has_hierarchical_mask(available_actions):
                action_log_probs, dist_entropy = self._evaluate_autoregressive(
                    x, action, available_actions, active_masks)
                # Collect per-dimension stats for TRPO compatibility
                mu_collector = []
                std_collector = []
                probs_collector = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    mu_collector.append(action_logit.mean)
                    std_collector.append(action_logit.stddev)
                    probs_collector.append(action_logit.logits)
                action_mu = torch.cat(mu_collector, -1)
                action_std = torch.cat(std_collector, -1)
                all_probs = torch.cat(probs_collector, -1)
            else:
                action_log_probs = []
                dist_entropy = []
                mu_collector = []
                std_collector = []
                probs_collector = []
                for action_out, act in zip(self.action_outs, action):
                    action_logit = action_out(x)
                    mu = action_logit.mean
                    std = action_logit.stddev
                    action_log_probs.append(action_logit.log_probs(act))
                    mu_collector.append(mu)
                    std_collector.append(std)
                    probs_collector.append(action_logit.logits)
                    if active_masks is not None:
                        dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())
                action_mu = torch.cat(mu_collector,-1)
                action_std = torch.cat(std_collector,-1)
                all_probs = torch.cat(probs_collector,-1)
                action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
                dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
