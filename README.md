# CC4 Trainable Red Agent Environment

This folder is a curated CC4 training tree focused on the trainable red-agent implementation. It keeps the CybORG CC4 simulator, the red PPO training entry point, and the blue baseline needed as an opponent. Communication research variants and paper/analysis assets are intentionally left out.

The default red-agent implementation is `scripts/train/train_red_ppo.py`. It trains the red team with IPPO or MAPPO over the same CC4 red action semantics used by the original rule-based attacker.

Included:

- `CybORG/`: CC4 simulator, agents, wrappers, actions, state, scenario generation, and rewards.
- `Hierarchical-MARL-main/marl-1policy/`: trainable blue PPO pipeline and blue policy loader.
- `scripts/train/train_red_ppo.py`: red PPO training entry with IPPO and MAPPO modes.
- `on-policy/`: on-policy MAPPO/IPPO dependency used by the red scripts.
- `scripts/launch/`: thin launchers for blue and red training.

Not included:

- HiComm, CACom, T2MAC, naive learned communication, communication ablations, papers, slides, analysis notebooks/scripts, and evaluation research harnesses.

## Red Agent Implementation

The trainable red agent is derived from the original CC4 finite-state red attacker. The original `FiniteStateRedAgent` chooses hand-coded actions from a finite-state progression: discover remote systems, scan services, exploit, escalate, degrade, withdraw, and impact. This implementation keeps that action vocabulary and the same CybORG action execution path, but replaces the fixed FSM decision rule with a neural PPO policy.

The conversion has three main pieces:

- `RedFixedActionWrapper` maps a compact factorized action tuple `(action_type, target_subnet, target_host)` back to concrete CybORG red actions. It also builds validity masks from the red agent's discovered knowledge, keeping the trainable policy close to the original FSM action constraints.
- `RedFlatWrapper` turns each red agent's local CybORG observation and belief state into a flat vector that can be consumed by the policy network.
- `train_red_ppo.py` wires the wrapped CC4 environment into the on-policy IPPO/MAPPO runner, saves checkpoints, and optionally trains against either sleeping blue agents or a trained blue MARL checkpoint.

In short: the FSM's hard-coded policy is removed, but its environment-facing mechanics are preserved. The learned policy now decides which FSM-style red action to take and where to target it.

### FSM State to RL Observation

The original FSM stores a per-host symbolic state and updates it after each action:

```text
K/KD = known host
S/SD = service-scanned host
U/UD = user-level access
R/RD = root-level access
F    = filtered or outside the agent's usable area
D suffix = neighbors/subnet expansion already discovered
```

The RL agent receives the same progression as numeric features in `RedFlatWrapper`. Each red agent has its own local belief matrix:

```text
9 subnets x 16 host slots x 5 features = 720 host features
```

The five per-host features are:

| Feature | Meaning | FSM correspondence |
|---|---|---|
| `access_level` | `0=none`, `1=user`, `2=root` | `U/UD -> 1`, `R/RD -> 2` |
| `recon_level` | `0=unknown`, `1=known`, `2=scanned` | `K/KD -> 1`, `S/SD -> 2` |
| `neighbors_discovered` | subnet expansion already done | FSM `D` suffix |
| `is_reachable` | whether recent evidence says the host can be reached | failed actions or blocking can set this to `0` |
| `deception_known` | whether deception was discovered on that host | successful `DiscoverDeception` |

The full red observation is:

```text
[mission_phase, last_action_success] + host_matrix + reserved_message_slots
= 2 + 720 + 80 = 802 floats
```

The 80 message-slot floats are kept for observation-shape compatibility with the red wrappers, but `train_red_ppo.py` does not train a communication policy, so they are normally zero.

### FSM Action to RL Action

The FSM picks a host, then samples one valid CybORG red action from its state-transition probability table. The RL version instead samples a factorized action:

```text
MultiDiscrete([action_type, target_subnet, target_host])
```

`RedFixedActionWrapper` translates that tuple back into the concrete CybORG action object. `target_host=0` is reserved for subnet-level or no-target actions; host-targeted actions require `target_host>0`.

| RL `action_type` | Concrete CybORG action | Target |
|---:|---|---|
| 0 | `DiscoverRemoteSystems` | subnet |
| 1 | `StealthServiceDiscovery` | host |
| 2 | `AggressiveServiceDiscovery` | host |
| 3 | `ExploitRemoteService` | host |
| 4 | `DegradeServices` | host |
| 5 | `DiscoverDeception` | host |
| 6 | `Withdraw` | host |
| 7 | `PrivilegeEscalate` | host |
| 8 | `Impact` | host |
| 9 | `Sleep` | none |

The action semantics come from the FSM attacker, but the RL index order is the wrapper's stable API, not the original FSM transition-matrix column order.

Action masks preserve the FSM-style progression:

| Belief condition | Allowed host actions |
|---|---|
| known only (`recon_level=1`) | stealth scan, aggressive scan |
| service scanned (`recon_level=2`) | exploit, discover deception |
| user access (`access_level=1`) | privilege escalate, withdraw |
| root access (`access_level=2`) | degrade services, impact, withdraw |
| neighbors not discovered | `DiscoverRemoteSystems` may also be enabled |
| inactive or no valid target | `Sleep` |

## Training Shape

Red actions are represented as `MultiDiscrete([action_type, subnet, host])`, with hierarchical masks so the policy only samples structurally valid targets. PPO computes the log probability of the full joint action, not independent per-branch updates.

IPPO and MAPPO are both enabled:

- IPPO uses separate red policies by agent id.
- MAPPO uses a shared red policy and a centralized critic over all red agents' observations, while each actor still receives only its own local observation.

## Reward

The red training reward is computed in `HybridFixedBlueWrapper` from the blue team's step reward. After the combined red/blue environment step:

1. The wrapper collects each active blue agent's scalar reward.
2. For each active red agent, it finds blue agents whose allowed subnet set overlaps that red agent's allowed subnet set.
3. The red agent receives the negative mean of those overlapping blue rewards.
4. If no blue subnet overlaps, it falls back to the negative mean over all blue rewards.

So the reward is dense and zero-sum in sign:

```text
r_red(agent_i) = -mean(r_blue(overlapping defenders))
```

Higher red reward means the red policy caused worse blue outcome. This is also why eval reports red reward with "higher is better for red". The evaluator additionally reports successful `Impact` counts, because reward alone can improve through non-objective actions such as degradation; `Impact` is the mission-critical red objective.

## Install

From this folder:

```bash
pip install -r Requirements.txt
pip install -e .
cd on-policy && pip install -e . && cd ..
```

## Train Blue

```bash
bash scripts/launch/train_blue.sh
```

Useful overrides:

```bash
BLUE_NUM_ITERATIONS=1 BLUE_NUM_ROLLOUT_WORKERS=0 BLUE_TRAIN_BATCH_SIZE=4000 bash scripts/launch/train_blue.sh
```

Blue checkpoints are written under:

```text
Hierarchical-MARL-main/marl-1policy/models/train_marl/
```

## Train Red

The red launchers default to `BLUE_OPPONENT=sleep`, so red training works before a blue checkpoint exists:

```bash
bash scripts/launch/train_red_ppo_ippo.sh
bash scripts/launch/train_red_ppo_mappo.sh
```

To train red against a trained blue policy:

```bash
BLUE_OPPONENT=marl BLUE_CHECKPOINT_ITER=iter_199 bash scripts/launch/train_red_ppo_ippo.sh
```

If your blue checkpoint lives outside the default model root, set:

```bash
BLUE_CHECKPOINT_ROOT=/path/to/models/train_marl BLUE_CHECKPOINT_ITER=iter_199 bash scripts/launch/train_red_ppo_ippo.sh
```

Red checkpoints are written under `red_checkpoints/ppo/{ippo,mappo}/`.

## Evaluate Red

Evaluation uses `scripts/eval/eval_red_ppo.py`. This is the clean-CC4 version of the parent repository's eval harness, narrowed to the `train_red_ppo.py` checkpoint format. It loads the saved red actor weights from `red_checkpoints/ppo/<alg>/<tag>/`, rebuilds the same wrapped CC4 red environment, and runs full episodes against a selected blue opponent.

```bash
bash scripts/launch/eval_red_ppo_ippo.sh
bash scripts/launch/eval_red_ppo_mappo.sh
```

The launchers default to:

```text
RED_POLICY_TAG=best
EPISODES=100
BLUE_OPPONENT=marl
BLUE_CHECKPOINT_ROOT=Hierarchical-MARL-main/marl-1policy/models/train_marl
BLUE_CHECKPOINT_ITER=iter_199
```

For a smoke eval without a trained blue checkpoint:

```bash
BLUE_OPPONENT=sleep EPISODES=1 bash scripts/launch/eval_red_ppo_ippo.sh
```

Direct invocation also works:

```bash
python scripts/eval/eval_red_ppo.py \
  --RL_TRAIN_ALG ippo \
  --red-policy-dir red_checkpoints/ppo/ippo/best \
  --blue-opponent marl \
  --episodes 100 \
  --output-dir evaluation_output/red_ppo_ippo_best
```

The evaluator loads either per-agent IPPO/MAPPO actors (`actor_agent_0.pt` ... `actor_agent_5.pt`) or a shared actor (`actor.pt`) based on `runner_meta.json`. Actions are served through the same `MultiDiscrete([action_type, subnet, host])` actor used in training, with the same hierarchical action masks from the environment. By default evaluation is deterministic, using the actor's mode; pass `--stochastic` to sample actions.

Eval outputs are written to the selected output directory:

- `summary.json`: aggregate red reward, reward std/min/max, Impact attempts, successful Impacts, Impact success rate, and action counts.
- `summary.txt`: compact human-readable summary.
- `episodes.json`: per-episode reward and Impact totals.
- `red_log.json`: per-step red action tuple, action name, target, success bit, and reward.

Impact success is counted when the selected action type is `Impact` and the next red observation reports `success=1`. Red reward is summed across all red agents and uses the same zero-sum signal as training: higher red reward means worse blue outcome.
