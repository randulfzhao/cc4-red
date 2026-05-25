from __future__ import annotations
from gymnasium import Space, spaces

from CybORG import CybORG
from CybORG.Agents.Wrappers import BaseWrapper
from CybORG.Simulator.Actions import Action, Sleep
from CybORG.Simulator.Actions.AbstractActions import (
    DiscoverRemoteSystems,
    StealthServiceDiscovery,
    AggressiveServiceDiscovery,
    ExploitRemoteService,
    PrivilegeEscalate,
    Impact,
    DiscoverDeception,
    DegradeServices,
)
from CybORG.Simulator.Actions.ConcreteActions import Withdraw
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)

import functools
import numpy as np
from typing import Any

# Define string formats based on ScenarioGenerator
SUBNET_USER_FORMAT = "{subnet}_user_host_{host}"
SUBNET_SERVER_FORMAT = "{subnet}_server_host_{host}"
SUBNET_ROUTER_FORMAT = "{subnet}_router"

MESSAGE_LENGTH = 16
EMPTY_MESSAGE = np.zeros(MESSAGE_LENGTH, dtype=np.float32)

DISABLE_SANITY_CHECKS = True

ACTION_TYPE_MAP = {
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

class RedFixedActionWrapper(BaseWrapper):
    """Maps factorized (action_type, target) tuples to CybORG Action objects."""

    def __init__(self, env: CybORG, *args, **kwargs):
        super().__init__(env)

        # Filter for red agents
        self.agents = self.possible_agents = [a for a in env.agents if "red" in a]

        self._agent_metadata = {}
        self._knowledge = {}
        self._host_state = {}
        self._action_masks = {}

        self._subnets = [
            "restricted_zone_a_subnet", "operational_zone_a_subnet",
            "restricted_zone_b_subnet", "operational_zone_b_subnet",
            "contractor_network_subnet", "public_access_zone_subnet",
            "admin_network_subnet", "office_network_subnet", "internet_subnet"
        ]
        self._subnet_to_index = {name: idx for idx, name in enumerate(self._subnets)}
        self._max_user_hosts = EnterpriseScenarioGenerator.MAX_USER_HOSTS
        self._max_server_hosts = EnterpriseScenarioGenerator.MAX_SERVER_HOSTS
        self._hosts_per_subnet = self._max_user_hosts + self._max_server_hosts
        self._host_index_range = self._hosts_per_subnet + 1  # 0 is subnet-level target

        for agent in self.agents:
            self._create_hardcoded_metadata(agent)
            self._init_knowledge(agent)
            self._reset_host_state(agent)

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, dict]]:
        reset_result = self.env.reset(*args, **kwargs)
        for agent in self.possible_agents:
            self._init_knowledge(agent)
            self._reset_host_state(agent)

        returned_obs = {}
        if (
            isinstance(reset_result, tuple)
            and len(reset_result) >= 1
            and isinstance(reset_result[0], dict)
        ):
            returned_obs = {
                agent: obs
                for agent, obs in reset_result[0].items()
                if isinstance(agent, str) and "red" in agent
            }

        self.agents = self._current_active_red_agents(returned_obs)
        observations = {}
        for agent in self.agents:
            if agent in returned_obs:
                observations[agent] = returned_obs[agent]
            else:
                observations[agent] = self.env.get_observation(agent)

        for agent_name, obs in observations.items():
            self._update_knowledge(agent_name, obs)
            self._update_action_mask(agent_name)
        info = {a: {"action_mask": self._action_masks[a]} for a in self.agents}
        return observations, info

    def step(
        self,
        actions: dict[str, Any] = None,
        messages: dict[str, Any] = None,
        **kwargs,
    ) -> tuple[dict, dict, dict, dict, dict]:

        action_dict = {} if actions is None else actions

        translated_actions = {}
        extracted_messages = {}
        for agent, action in action_dict.items():
            # Dict action space: separate action and message
            if isinstance(action, dict) and "action" in action:
                if "message" in action:
                    extracted_messages[agent] = np.clip(
                        np.array(action["message"], dtype=np.float32), -1.0, 1.0
                    )
                action = action["action"]
            if isinstance(action, Action):
                translated_actions[agent] = action
            else:
                translated_actions[agent] = self._translate_factorized_action(agent, action)

        # Merge extracted messages with any explicitly passed messages; default to zeros
        messages = {} if messages is None else messages
        messages = {
            agent: np.clip(
                np.array(
                    extracted_messages.get(agent, messages.get(agent, EMPTY_MESSAGE)),
                    dtype=np.float32,
                ),
                -1.0, 1.0,
            )
            for agent in self.possible_agents
        }

        # If wrapped around a higher-level red-vs-blue wrapper stack, call step()
        # so Blue policy injection/reward shaping wrappers are not bypassed.
        if isinstance(self.env, CybORG):
            obs, rews, dones, info = self.env.parallel_step(
                translated_actions, messages=messages, **kwargs
            )
            terminated = {
                agent: done for agent, done in dones.items() if "red" in agent
            }
            truncated = {
                agent: done for agent, done in dones.items() if "red" in agent
            }
        else:
            step_result = self.env.step(translated_actions, messages=messages, **kwargs)
            if len(step_result) == 5:
                obs, rews, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                obs, rews, dones, info = step_result
                terminated = {
                    agent: done for agent, done in dones.items() if "red" in agent
                }
                truncated = {
                    agent: done for agent, done in dones.items() if "red" in agent
                }
            else:
                raise ValueError(
                    f"Unexpected step return signature for {type(self.env)}: {len(step_result)}"
                )

        active_red_agents = self._current_active_red_agents(obs)
        self.agents = active_red_agents

        for agent_name in self.agents:
            if agent_name in obs:
                self._update_knowledge(
                    agent_name, obs[agent_name],
                    submitted_action=translated_actions.get(agent_name),
                )
            self._update_action_mask(agent_name)

        observations = {agent: o for agent, o in obs.items() if "red" in agent}
        observations = {
            agent: o
            for agent, o in observations.items()
            if agent in self.agents
        }
        rewards = {
            agent: self._sum_reward_components(reward)
            for agent, reward in rews.items()
            if "red" in agent
        }
        terminated = {
            agent: done
            for agent, done in terminated.items()
            if "red" in agent or agent == "__all__"
        }
        truncated = {
            agent: done
            for agent, done in truncated.items()
            if "red" in agent or agent == "__all__"
        }
        if not isinstance(info, dict):
            info = {}
        else:
            info = {
                agent: agent_info
                for agent, agent_info in info.items()
                if isinstance(agent, str) and "red" in agent and isinstance(agent_info, dict)
            }
        for agent_name in self.possible_agents:
            existing = info.get(agent_name, {})
            if not isinstance(existing, dict):
                existing = {}
            if agent_name in self.agents:
                existing["action_mask"] = self._action_masks[agent_name]
            else:
                existing["action_mask"] = self._sleep_action_mask()
            info[agent_name] = existing

        return observations, rewards, terminated, truncated, info

    def _current_active_red_agents(self, observations: dict[str, Any] | None = None) -> list[str]:
        """Return active red agents in canonical order.

        Some wrappers return only active observations; the underlying CybORG
        interface is still the source of truth when available.
        """
        observed = set(observations or {})
        controller = getattr(self.env, "environment_controller", None)
        interfaces = getattr(controller, "agent_interfaces", {})
        active = []
        for agent in self.possible_agents:
            interface = interfaces.get(agent)
            if interface is not None:
                if interface.active:
                    active.append(agent)
            elif agent in observed:
                active.append(agent)
        return active

    def _sum_reward_components(self, reward: Any) -> float:
        if isinstance(reward, dict):
            return float(sum(reward.values()))
        if reward is None:
            return 0.0
        try:
            return float(reward)
        except (TypeError, ValueError):
            return 0.0

    def _update_action_mask(self, agent_name: str) -> None:
        """
        Updates the action mask based on the agent's current belief state.
        """
        host_state = self._host_state[agent_name]
        n_action = len(ACTION_TYPE_MAP)
        n_subnet = len(self._subnets)
        n_host = self._host_index_range

        action_type_mask = np.zeros(n_action, dtype=np.float32)
        subnet_by_action = np.zeros((n_action, n_subnet), dtype=np.float32)
        host_by_action_subnet = np.zeros((n_action, n_subnet, n_host), dtype=np.float32)

        for subnet_idx in range(n_subnet):
            for host_slot in range(self._hosts_per_subnet):
                access_level = host_state[subnet_idx, host_slot, 0]
                recon_level = host_state[subnet_idx, host_slot, 1]
                neighbors = host_state[subnet_idx, host_slot, 2]
                allowed = self._allowed_action_types_for_state(
                    access_level, recon_level, neighbors
                )
                if not allowed:
                    continue
                for action_type in allowed:
                    action_type_mask[action_type] = 1.0
                    if action_type == 0:
                        # DiscoverRemoteSystems: subnet-level target.
                        subnet_by_action[action_type, subnet_idx] = 1.0
                        host_by_action_subnet[action_type, subnet_idx, 0] = 1.0
                    elif action_type in (1, 2, 3, 4, 5, 6, 7, 8):
                        # Host-targeted actions: concrete host index required (>0).
                        host_idx = host_slot + 1
                        subnet_by_action[action_type, subnet_idx] = 1.0
                        host_by_action_subnet[action_type, subnet_idx, host_idx] = 1.0

        # --- Enable DRS for undiscovered subnets that CybORG already knows about ---
        # Only allow DRS on subnets whose CIDR is in the agent's ActionSpace,
        # otherwise CybORG's replace_action_if_invalid will reject the action.
        state = self.env.environment_controller.state
        agent_iface = self.env.environment_controller.agent_interfaces.get(agent_name)
        known_subnets = agent_iface.action_space.subnet if agent_iface else {}
        for subnet_idx in range(n_subnet):
            has_known_host = False
            for host_slot in range(self._hosts_per_subnet):
                if np.max(host_state[subnet_idx, host_slot]) > 0:
                    has_known_host = True
                    break
            if not has_known_host:
                subnet_name = self._subnets[subnet_idx]
                subnet_cidr = state.subnet_name_to_cidr.get(subnet_name)
                if subnet_cidr is not None and known_subnets.get(subnet_cidr, False):
                    action_type_mask[0] = 1.0
                    subnet_by_action[0, subnet_idx] = 1.0
                    host_by_action_subnet[0, subnet_idx, 0] = 1.0

        # Sleep is always allowed and deterministically maps to (subnet=0, host=0).
        action_type_mask[9] = 1.0
        subnet_by_action[9, 0] = 1.0
        host_by_action_subnet[9, 0, 0] = 1.0

        # Marginal masks used by the policy model for coarse invalid-action masking.
        subnet_mask = (subnet_by_action.max(axis=0) > 0).astype(np.float32)
        host_mask = (host_by_action_subnet.max(axis=(0, 1)) > 0).astype(np.float32)

        # Always preserve no-target index for deterministic subnet/no-target actions.
        subnet_mask[0] = 1.0
        host_mask[0] = 1.0

        self._action_masks[agent_name] = {
            "action_type": action_type_mask,
            "target_subnet": subnet_mask,
            "target_host": host_mask,
            "target_subnet_by_action": subnet_by_action,
            "target_host_by_action_subnet": host_by_action_subnet,
        }

    def _sleep_action_mask(self) -> dict[str, np.ndarray]:
        n_action = len(ACTION_TYPE_MAP)
        n_subnet = len(self._subnets)
        n_host = self._host_index_range
        action_type_mask = np.zeros(n_action, dtype=np.float32)
        subnet_mask = np.zeros(n_subnet, dtype=np.float32)
        host_mask = np.zeros(n_host, dtype=np.float32)
        subnet_by_action = np.zeros((n_action, n_subnet), dtype=np.float32)
        host_by_action_subnet = np.zeros((n_action, n_subnet, n_host), dtype=np.float32)

        action_type_mask[9] = 1.0
        subnet_mask[0] = 1.0
        host_mask[0] = 1.0
        subnet_by_action[9, 0] = 1.0
        host_by_action_subnet[9, 0, 0] = 1.0
        return {
            "action_type": action_type_mask,
            "target_subnet": subnet_mask,
            "target_host": host_mask,
            "target_subnet_by_action": subnet_by_action,
            "target_host_by_action_subnet": host_by_action_subnet,
        }

    def _init_knowledge(self, agent_name: str) -> None:
        self._knowledge[agent_name] = {
            "known_hosts": set(),
            "known_subnets": set(),
        }

    def _update_knowledge(self, agent_name: str, observation: dict[str, Any],
                          submitted_action: Action | None = None) -> None:
        """
        Maintain a light-weight belief state of what the red agent has actually seen.
        This avoids masking purely by ground truth and keeps the policy closer to the FSM logic.

        Parameters
        ----------
        submitted_action : Action | None
            The CybORG Action object that was submitted for this agent in the
            current step.  We use this instead of ``observation.get("action")``
            because ``Observation.combine_obs()`` strips the ``"action"`` key
            when merging observations, so the observation dict never contains it.
        """
        if agent_name not in self._knowledge:
            return

        state = self.env.environment_controller.state
        knowledge = self._knowledge[agent_name]

        # --- Session sync: regress access_level for hosts that lost sessions ---
        active_session_hosts = set()
        for session in state.sessions.get(agent_name, {}).values():
            if session.active:
                active_session_hosts.add(session.hostname)

        for subnet_idx in range(len(self._subnets)):
            for host_slot in range(self._hosts_per_subnet):
                feats = self._host_state[agent_name][subnet_idx, host_slot]
                if feats[0] > 0:
                    hostname = self._hostname_from_target(
                        self._subnets[subnet_idx], host_slot + 1
                    )
                    if hostname is not None and hostname not in active_session_hosts:
                        feats[0] = 0.0

        # Use the submitted action rather than observation.get("action"),
        # because Observation.combine_obs() discards the "action" key.
        action = submitted_action or observation.get("action")
        action_success = self._obs_success(observation)

        if action is not None and action_success and isinstance(action, DiscoverRemoteSystems):
            subnet_idx = self._resolve_subnet_index(getattr(action, "subnet", None))
            if subnet_idx is not None:
                host_block = self._host_state[agent_name][subnet_idx]
                host_block[:, 1] = np.maximum(host_block[:, 1], 1.0)
                host_block[:, 2] = 1.0
                host_block[:, 3] = 1.0
                knowledge["known_subnets"].add(self._subnets[subnet_idx])

        if action is not None:
            if isinstance(action, (StealthServiceDiscovery, AggressiveServiceDiscovery)):
                hostname = self._resolve_hostname_by_ip(getattr(action, "ip_address", None))
                if action_success:
                    self._set_host_feature(agent_name, hostname, recon_level=2.0, reachable=1.0)
                else:
                    self._set_host_feature(agent_name, hostname, reachable=0.0)
            elif isinstance(action, ExploitRemoteService):
                hostname = self._resolve_hostname_by_ip(getattr(action, "ip_address", None))
                if action_success:
                    self._set_host_feature(agent_name, hostname, access_level=1.0, reachable=1.0)
                else:
                    self._set_host_feature(agent_name, hostname, reachable=0.0)
            elif isinstance(action, PrivilegeEscalate):
                hostname = getattr(action, "hostname", None)
                if action_success:
                    self._set_host_feature(agent_name, hostname, access_level=2.0, reachable=1.0)
                else:
                    self._set_host_feature(agent_name, hostname, reachable=0.0)
            elif isinstance(action, DiscoverDeception):
                hostname = self._resolve_hostname_by_ip(getattr(action, "ip_address", None))
                if action_success:
                    self._set_host_feature(agent_name, hostname, deception_known=1.0, reachable=1.0)
                else:
                    self._set_host_feature(agent_name, hostname, reachable=0.0)
            elif isinstance(action, (DegradeServices, Impact, Withdraw)):
                hostname = getattr(action, "hostname", None)
                if not action_success:
                    self._set_host_feature(agent_name, hostname, reachable=0.0)

        for session in state.sessions.get(agent_name, {}).values():
            if not session.active:
                continue
            access_level = 2.0 if session.username in ["root", "SYSTEM"] else 1.0
            self._set_host_feature(
                agent_name,
                session.hostname,
                access_level=access_level,
                recon_level=1.0,
                reachable=1.0,
            )

        for key, value in observation.items():
            if key in {"success", "action", "message"}:
                continue
            if not isinstance(value, dict):
                continue

            hostname = None
            if key in state.hosts:
                hostname = key
            else:
                hostname = self._resolve_hostname_by_ip(key)

            if hostname is None:
                interfaces = value.get("Interface") or value.get("Interfaces")
                if interfaces:
                    for iface in interfaces:
                        ip_val = iface.get("ip_address") or iface.get("IP Address")
                        if ip_val is None:
                            continue
                        resolved = self._resolve_hostname_by_ip(ip_val)
                        if resolved:
                            hostname = resolved
                            break

            if hostname is None:
                continue

            knowledge["known_hosts"].add(hostname)
            self._set_host_feature(agent_name, hostname, recon_level=1.0, reachable=1.0)

            if "Processes" in value:
                self._set_host_feature(agent_name, hostname, recon_level=2.0, reachable=1.0)
                if self._process_has_decoy(value.get("Processes", [])):
                    self._set_host_feature(agent_name, hostname, deception_known=1.0)

            if "Sessions" in value:
                access = self._access_from_sessions(value["Sessions"])
                if access > 0:
                    self._set_host_feature(agent_name, hostname, access_level=float(access), reachable=1.0)

    def _resolve_hostname_by_ip(self, ip_value: Any) -> str | None:
        """
        Resolve a hostname from an IP value that might be an IPv4Address or a string.
        """
        state = self.env.environment_controller.state
        ip_str = str(ip_value)
        for h_name, h_obj in state.hosts.items():
            for interface in h_obj.interfaces:
                if str(interface.ip_address) == ip_str:
                    return h_name
        return None

    def _create_hardcoded_metadata(self, agent_name: str) -> None:
        """Identifies all hosts and subnets that the agent will ever encounter."""
        hosts = set()
        for subnet in self._subnets:
            if subnet == "internet_subnet":
                hosts.add("root_internet_host_0")
                continue
            for i in range(self._max_user_hosts):
                hosts.add(SUBNET_USER_FORMAT.format(subnet=subnet, host=i))
            for i in range(self._max_server_hosts):
                hosts.add(SUBNET_SERVER_FORMAT.format(subnet=subnet, host=i))

        self._agent_metadata[agent_name] = {
            "hosts": sorted(hosts),
            "subnets": list(self._subnets),
            "host_to_subnet": self._build_host_to_subnet_map(),
        }

    def _host_sanity_check(self) -> None:
        pass

    def _build_host_to_subnet_map(self) -> dict[str, str]:
        host_to_subnet: dict[str, str] = {}
        for subnet in self._subnets:
            if subnet == "internet_subnet":
                host_to_subnet["root_internet_host_0"] = subnet
                continue
            for i in range(self._max_user_hosts):
                hostname = SUBNET_USER_FORMAT.format(subnet=subnet, host=i)
                host_to_subnet[hostname] = subnet
            for i in range(self._max_server_hosts):
                hostname = SUBNET_SERVER_FORMAT.format(subnet=subnet, host=i)
                host_to_subnet[hostname] = subnet
        return host_to_subnet

    def _reset_host_state(self, agent_name: str) -> None:
        self._host_state[agent_name] = np.zeros(
            (len(self._subnets), self._hosts_per_subnet, 5), dtype=np.float32
        )

    def _allowed_action_types_for_state(self, access_level, recon_level, neighbors_discovered):
        if access_level >= 2:
            allowed = {4, 8, 6}
            if neighbors_discovered < 1:
                allowed.add(0)
            return allowed
        if access_level >= 1:
            allowed = {7, 6}
            if neighbors_discovered < 1:
                allowed.add(0)
            return allowed
        if recon_level >= 2:
            allowed = {5, 3}
            if neighbors_discovered < 1:
                allowed.add(0)
            return allowed
        if recon_level >= 1:
            allowed = {1, 2}
            if neighbors_discovered < 1:
                allowed.add(0)
            return allowed
        return set()

    def _set_host_feature(
        self,
        agent_name,
        hostname,
        access_level=None,
        recon_level=None,
        neighbors_discovered=None,
        reachable=None,
        deception_known=None,
    ):
        if not hostname:
            return
        subnet = self._agent_metadata[agent_name]["host_to_subnet"].get(hostname)
        if subnet is None:
            return
        subnet_idx = self._subnet_to_index.get(subnet)
        if subnet_idx is None:
            return
        host_slot = self._host_slot_from_hostname(hostname)
        if host_slot is None:
            return
        feats = self._host_state[agent_name][subnet_idx, host_slot]
        if access_level is not None:
            feats[0] = max(feats[0], float(access_level))
        if recon_level is not None:
            feats[1] = max(feats[1], float(recon_level))
        if neighbors_discovered is not None:
            feats[2] = max(feats[2], float(neighbors_discovered))
        if reachable is not None:
            feats[3] = max(0.0, min(1.0, float(reachable)))
        if deception_known is not None:
            feats[4] = max(feats[4], float(deception_known))

    def _host_slot_from_hostname(self, hostname: str) -> int | None:
        if hostname == "root_internet_host_0":
            return self._max_user_hosts
        if "_user_host_" in hostname:
            try:
                return int(hostname.split("_user_host_")[-1])
            except ValueError:
                return None
        if "_server_host_" in hostname:
            try:
                return self._max_user_hosts + int(hostname.split("_server_host_")[-1])
            except ValueError:
                return None
        return None

    def _translate_factorized_action(self, agent_name: str, action: Any) -> Action:
        action_type, subnet_idx, host_idx = self._parse_action_tuple(action)

        if action_type == 9:
            return Sleep()

        session_id = self._active_session_id(agent_name)
        if session_id is None:
            return Sleep()

        if subnet_idx is None or subnet_idx < 0 or subnet_idx >= len(self._subnets):
            return Sleep()

        subnet_name = self._subnets[subnet_idx]
        state = self.env.environment_controller.state

        if action_type == 0:
            if not self._subnet_allows_drs(agent_name, subnet_idx):
                return Sleep()
            subnet_cidr = state.subnet_name_to_cidr.get(subnet_name)
            if subnet_cidr is None:
                return Sleep()
            return DiscoverRemoteSystems(subnet=subnet_cidr, agent=agent_name, session=session_id)

        if host_idx is None or host_idx <= 0:
            return Sleep()

        hostname = self._hostname_from_target(subnet_name, host_idx)
        if hostname is None:
            return Sleep()

        if not self._is_action_allowed_for_host(agent_name, action_type, subnet_idx, host_idx):
            return Sleep()

        if hostname not in state.hosts:
            return Sleep()

        target_ip = state.hosts[hostname].interfaces[0].ip_address

        if action_type == 1:
            return StealthServiceDiscovery(session=session_id, agent=agent_name, ip_address=target_ip)
        if action_type == 2:
            return AggressiveServiceDiscovery(session=session_id, agent=agent_name, ip_address=target_ip)
        if action_type == 3:
            return ExploitRemoteService(ip_address=target_ip, session=session_id, agent=agent_name)
        if action_type == 4:
            return DegradeServices(hostname=hostname, session=session_id, agent=agent_name)
        if action_type == 5:
            return DiscoverDeception(session=session_id, agent=agent_name, ip_address=target_ip)
        if action_type == 6:
            return Withdraw(session=session_id, agent=agent_name, ip_address=target_ip, hostname=hostname)
        if action_type == 7:
            return PrivilegeEscalate(hostname=hostname, session=session_id, agent=agent_name)
        if action_type == 8:
            return Impact(hostname=hostname, session=session_id, agent=agent_name)

        return Sleep()

    def _active_session_id(self, agent_name: str) -> int | None:
        state = self.env.environment_controller.state
        sessions = state.sessions.get(agent_name, {})

        session_zero = sessions.get(0)
        if session_zero is not None and session_zero.active:
            return 0

        for session_id in sorted(sessions):
            session = sessions[session_id]
            if session.active:
                return int(session_id)

        return None

    def _resolve_subnet_index(self, subnet: Any) -> int | None:
        if subnet is None:
            return None

        if isinstance(subnet, str):
            return self._subnet_to_index.get(subnet)

        state = self.env.environment_controller.state
        for subnet_name, subnet_cidr in state.subnet_name_to_cidr.items():
            if subnet_cidr == subnet:
                return self._subnet_to_index.get(subnet_name)

        return None

    def _parse_action_tuple(self, action: Any) -> tuple[int, int | None, int | None]:
        if isinstance(action, dict):
            action_type = int(action.get("action_type", 9))
            target = action.get("target", None)
            if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 2:
                return action_type, int(target[0]), int(target[1])
            return action_type, None, None
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 3:
                return int(action[0]), int(action[1]), int(action[2])
            if len(action) == 2:
                action_type = int(action[0])
                target = action[1]
                if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 2:
                    return action_type, int(target[0]), int(target[1])
                return action_type, None, None
        return 9, None, None

    def _hostname_from_target(self, subnet_name: str, host_idx: int) -> str | None:
        if host_idx <= 0:
            return None
        host_slot = host_idx - 1
        if subnet_name == "internet_subnet":
            if host_slot == self._max_user_hosts:
                return "root_internet_host_0"
            return None
        if host_slot < self._max_user_hosts:
            return SUBNET_USER_FORMAT.format(subnet=subnet_name, host=host_slot)
        server_slot = host_slot - self._max_user_hosts
        if 0 <= server_slot < self._max_server_hosts:
            return SUBNET_SERVER_FORMAT.format(subnet=subnet_name, host=server_slot)
        return None

    def _is_action_allowed_for_host(self, agent_name: str, action_type: int, subnet_idx: int, host_idx: int) -> bool:
        host_slot = host_idx - 1
        if host_slot < 0 or host_slot >= self._hosts_per_subnet:
            return False
        feats = self._host_state[agent_name][subnet_idx, host_slot]
        allowed = self._allowed_action_types_for_state(feats[0], feats[1], feats[2])
        return action_type in allowed

    def _subnet_allows_drs(self, agent_name: str, subnet_idx: int) -> bool:
        has_known_host = False
        for host_slot in range(self._hosts_per_subnet):
            feats = self._host_state[agent_name][subnet_idx, host_slot]
            if np.max(feats) > 0:
                has_known_host = True
            allowed = self._allowed_action_types_for_state(feats[0], feats[1], feats[2])
            if 0 in allowed:
                return True
        # For completely undiscovered subnets, only allow DRS if the subnet
        # is already in CybORG's ActionSpace (otherwise CybORG rejects it).
        if not has_known_host:
            state = self.env.environment_controller.state
            agent_iface = self.env.environment_controller.agent_interfaces.get(agent_name)
            if agent_iface is not None:
                subnet_name = self._subnets[subnet_idx]
                subnet_cidr = state.subnet_name_to_cidr.get(subnet_name)
                if subnet_cidr is not None and agent_iface.action_space.subnet.get(subnet_cidr, False):
                    return True
        return False

    def _obs_success(self, observation: dict[str, Any]) -> bool:
        success = observation.get("success", False)
        try:
            return bool(success.value == 1)
        except AttributeError:
            return bool(success)

    def _access_from_sessions(self, sessions) -> int:
        access_level = 0
        for session in sessions:
            user = session.get("username", session.get("Username"))
            if user is None:
                continue
            if user in ["root", "SYSTEM"]:
                access_level = 2
                break
            access_level = max(access_level, 1)
        return access_level

    def _process_has_decoy(self, processes) -> bool:
        for proc in processes:
            props = proc.get("Properties") or proc.get("properties") or []
            if "decoy" in props:
                return True
        return False

    def get_action_space(self, agent: str) -> dict:
        return self.action_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name: str) -> Space:
        return spaces.Dict({
            "action": spaces.MultiDiscrete(
                [len(ACTION_TYPE_MAP), len(self._subnets), self._host_index_range]
            ),
            "message": spaces.Box(-1.0, 1.0, shape=(MESSAGE_LENGTH,), dtype=np.float32),
        })
