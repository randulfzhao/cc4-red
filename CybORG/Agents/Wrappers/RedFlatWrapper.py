from __future__ import annotations
import numpy as np
from gymnasium import spaces

from CybORG.Agents.Wrappers.RedFixedActionWrapper import (
    RedFixedActionWrapper,
    MESSAGE_LENGTH,
    EMPTY_MESSAGE,
)

# Fixed ordering of all red agents for deterministic message slot assignment
ALL_RED_AGENTS = [
    "red_agent_0", "red_agent_1", "red_agent_2",
    "red_agent_3", "red_agent_4", "red_agent_5",
]
MAX_RED_SENDERS = len(ALL_RED_AGENTS) - 1  # 5 slots per agent


class RedFlatWrapper(RedFixedActionWrapper):
    """
    Flattens the observation into a fixed-length vector for Red Agents (IDQN).
    Structure: Global Info (2) + Host Matrix (9 * 16 * 5).
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        
        self.num_host_features = 5
        self.observation_state = {} # Persistent belief state per agent
        self._observation_space = {}

        # Initialize observation space
        for agent in self.agents:
            # Global (2) + Subnets * Hosts * Features + received messages (MAX_RED_SENDERS * MESSAGE_LENGTH)
            vec_size = (
                2
                + len(self._subnets) * self._hosts_per_subnet * self.num_host_features
                + MAX_RED_SENDERS * MESSAGE_LENGTH
            )

            # Messages are in [-1, 1]; host features in [0, 2]; use unbounded Box for simplicity
            self._observation_space[agent] = spaces.Box(
                low=-1.0, high=2.0, shape=(vec_size,), dtype=np.float32
            )
            
            self.observation_state[agent] = {
                'hosts': None,  # uses self._host_state from base wrapper
                'success': False,
                'received_messages': {},  # sender_id -> message array
            }

    def observation_space(self, agent: str):
        return self._observation_space[agent]

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        
        # Reset internal belief state
        for agent in self.possible_agents:
            self.observation_state[agent] = {
                'hosts': None,
                'success': False,
                'received_messages': {},
            }
        
        # Process initial observation (often empty for Red, but good to be safe)
        return self._process_observations(obs), info

    def step(self, actions=None, messages=None, **kwargs):
        obs, rewards, terminated, truncated, info = super().step(actions, messages, **kwargs)
        
        # Update belief state and flatten
        flat_obs = self._process_observations(obs)
        
        return flat_obs, rewards, terminated, truncated, info

    def _process_observations(self, observations):
        flat_observations = {}
        for agent, obs in observations.items():
            if agent not in self.agents:
                continue
            
            # Update belief state based on new observation
            self._update_belief_state(agent, obs)
            
            # Flatten to vector
            flat_observations[agent] = self._flatten_observation(agent)
            
        return flat_observations

    def _update_belief_state(self, agent, observation):
        """
        Parses the dictionary observation and updates the agent's internal belief state.
        """
        # 1. Update Global Info
        self.observation_state[agent]['success'] = self._obs_success(observation)

        # 2. Update received messages: list of (sender_id, message) tuples
        raw_messages = observation.get("message", [])
        received = {}
        for entry in raw_messages:
            if isinstance(entry, tuple) and len(entry) == 2:
                sender_id, msg = entry
                received[sender_id] = np.array(msg, dtype=np.float32)
        self.observation_state[agent]['received_messages'] = received
        
        # Host belief state is updated in RedFixedActionWrapper._update_knowledge.

    def _flatten_observation(self, agent):
        """
        Converts the belief state into a 1D vector.
        """
        # 1. Global Info
        # Mission Phase (Normalized 0-1)
        # We need to access mission phase.
        mission_phase = self.env.environment_controller.state.mission_phase
        # State stores mission phase as int (0/1/2). Keep string fallback for compatibility.
        if isinstance(mission_phase, (int, np.integer, float)):
            mission_val = float(np.clip(mission_phase, 0.0, 2.0))
        else:
            phase_map = {'Preplanning': 0.0, 'MissionA': 1.0, 'MissionB': 2.0}
            mission_val = phase_map.get(str(mission_phase), 0.0)
        
        success_val = 1.0 if self.observation_state[agent]['success'] else 0.0
        
        vector = [mission_val, success_val]
        
        # 2. Host Matrix
        host_matrix = self._host_state[agent]
        for subnet_idx in range(len(self._subnets)):
            for host_idx in range(self._hosts_per_subnet):
                vector.extend(host_matrix[subnet_idx, host_idx])

        # 3. Received messages in fixed slots (one slot per possible sender, ordered by agent id)
        received = self.observation_state[agent].get('received_messages', {})
        for sender in ALL_RED_AGENTS:
            if sender == agent:
                continue  # skip self
            msg = received.get(sender, EMPTY_MESSAGE)
            vector.extend(np.array(msg, dtype=np.float32))

        return np.array(vector, dtype=np.float32)
