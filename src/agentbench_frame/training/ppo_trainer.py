"""
Self-contained PPO (Proximal Policy Optimization) Trainer.

Implements PPO with GAE, clipped objective, action masking, and multi-epoch updates.

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import os, time, json, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np; HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from agentbench_frame.env.base import BaseEnv, Observation
from agentbench_frame.agent.policy_network import ActorCritic, create_generals_policy
from agentbench_frame.agent.base import BaseAgent


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 1024
    n_epochs: int = 10
    mini_batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    eval_freq: int = 50_000
    eval_episodes: int = 10
    save_freq: int = 100_000
    log_dir: str = "./rl_logs"
    verbose: bool = True


@dataclass
class RolloutBatch:
    boards: Any = None
    globals: Any = None
    actions: Any = None
    log_probs: Any = None
    values: Any = None
    rewards: Any = None
    dones: Any = None
    action_masks: Any = None
    advantages: Any = None
    returns: Any = None


class PPOTrainer:
    """Self-contained PPO trainer for AgentBench environments.

    Usage:
        env = GeneralsEnv(mode=EnvMode.DIRECT)
        config = PPOConfig(total_timesteps=200_000)
        trainer = PPOTrainer(env, config)
        trainer.train()
    """

    def __init__(self, env: BaseEnv, config: Optional[PPOConfig] = None,
                 eval_opponent: Optional[BaseAgent] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install: pip install torch")
        if not HAS_NUMPY:
            raise ImportError("NumPy required. Install: pip install numpy")

        self.env = env
        self.config = config or PPOConfig()
        self.eval_opponent = eval_opponent
        self.num_actions = 1000

        self.policy = create_generals_policy(num_actions=self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

        self._timesteps = 0
        self._episodes = 0
        self._best_eval_score = -float("inf")
        self._metrics_history: List[Dict] = []
        self._current_obs = None

        os.makedirs(self.config.log_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        if self.config.verbose:
            n_params = sum(p.numel() for p in self.policy.parameters())
            print(f"PPO Trainer: device={self.device}, params={n_params:,}, "
                  f"actions={self.num_actions}, steps={self.config.total_timesteps:,}")

    # ========== Public API ==========

    def train(self) -> Dict[str, Any]:
        """Run the full PPO training loop."""
        self._timesteps = 0
        self._episodes = 0
        self._current_obs = self.env.reset()
        cfg = self.config
        start_time = time.time()

        print(f"\nTraining ({cfg.total_timesteps:,} steps)...")
        while self._timesteps < cfg.total_timesteps:
            batch = self._collect_rollout()
            metrics = self._update(batch)
            self._metrics_history.append({"timesteps": self._timesteps, **metrics})

            if cfg.verbose and self._episodes % 5 == 0:
                print(f"  step={self._timesteps:>8,} p_loss={metrics['policy_loss']:.4f} "
                      f"v_loss={metrics['value_loss']:.4f} H={metrics['entropy']:.4f} "
                      f"R={metrics['mean_reward']:.2f}")

            if cfg.eval_freq and self._timesteps % cfg.eval_freq < cfg.rollout_steps:
                score = self._evaluate()
                print(f"  >>> Eval @ {self._timesteps:,}: score={score:.3f}")
                if score > self._best_eval_score:
                    self._best_eval_score = score
                    self.save(os.path.join(cfg.log_dir, "best_policy.pt"))

            if cfg.save_freq and self._timesteps % cfg.save_freq < cfg.rollout_steps:
                self.save(os.path.join(cfg.log_dir, f"policy_{self._timesteps}.pt"))

        elapsed = time.time() - start_time
        self.save(os.path.join(cfg.log_dir, "final_policy.pt"))
        summary = {"total_timesteps": self._timesteps, "total_episodes": self._episodes,
                   "elapsed_seconds": elapsed, "best_eval_score": self._best_eval_score}
        with open(os.path.join(cfg.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Done. Best: {self._best_eval_score:.3f} in {elapsed:.0f}s")
        return summary

    def save(self, path: str):
        from agentbench_frame.agent.policy_network import save_policy
        save_policy(self.policy, path)

    def load(self, path: str):
        from agentbench_frame.agent.policy_network import load_policy
        self.policy = load_policy(path)
        self.policy.to(self.device)

    # ========== Rollout ==========

    def _collect_rollout(self) -> RolloutBatch:
        cfg = self.config
        boards_l, globals_l, actions_l, logps_l = [], [], [], []
        values_l, rewards_l, dones_l, masks_l = [], [], [], []

        for _ in range(cfg.rollout_steps):
            board_t, glob_t, mask_t = self._obs_to_tensors(self._current_obs)

            with torch.no_grad():
                aid, lp, val = self.policy.get_action(
                    board_t.unsqueeze(0).to(self.device),
                    glob_t.unsqueeze(0).to(self.device),
                    mask_t.unsqueeze(0).to(self.device) if mask_t is not None else None,
                )

            ga = self._id_to_game_action(aid)
            obs, reward, done, info = self.env.step(ga)
            self._timesteps += 1

            boards_l.append(board_t); globals_l.append(glob_t)
            actions_l.append(aid); logps_l.append(lp); values_l.append(val)
            rewards_l.append(reward); dones_l.append(done)
            masks_l.append(mask_t if mask_t is not None else torch.ones(self.num_actions))

            if done:
                self._episodes += 1
                self._current_obs = self.env.reset()
            else:
                self._current_obs = obs

            if self._timesteps >= cfg.total_timesteps:
                break

        n = len(boards_l)
        rewards_t = torch.tensor(rewards_l, dtype=torch.float32)
        values_t = torch.tensor(values_l, dtype=torch.float32)
        dones_t = torch.tensor(dones_l, dtype=torch.float32)

        adv, ret = self._compute_gae(rewards_t, values_t, dones_t)

        return RolloutBatch(
            boards=torch.stack(boards_l), globals=torch.stack(globals_l),
            actions=torch.tensor(actions_l, dtype=torch.long),
            log_probs=torch.tensor(logps_l, dtype=torch.float32),
            values=values_t, rewards=rewards_t, dones=dones_t,
            action_masks=torch.stack(masks_l),
            advantages=adv, returns=ret,
        )

    def _compute_gae(self, rewards, values, dones) -> Tuple:
        cfg = self.config
        n = len(rewards)
        adv = torch.zeros(n); ret = torch.zeros(n)
        gae = 0.0; nv = 0.0
        for t in range(n - 1, -1, -1):
            nnt = 1.0 - dones[t].item()
            next_val = nv if t == n - 1 else values[t + 1]
            delta = rewards[t] + cfg.gamma * next_val * nnt - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nnt * gae
            adv[t] = gae; ret[t] = gae + values[t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # ========== PPO Update ==========

    def _update(self, batch: RolloutBatch) -> Dict[str, float]:
        cfg = self.config; n = len(batch.actions)
        total_pl, total_vl, total_ent, n_upd = 0.0, 0.0, 0.0, 0

        for _ in range(cfg.n_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, cfg.mini_batch_size):
                idx = indices[start:start + cfg.mini_batch_size]
                mb = self._to_device(batch, idx)

                logits, values = self.policy(mb.boards, mb.globals, mb.action_masks)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_lp = dist.log_prob(mb.actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_lp - mb.log_probs)

                surr1 = ratio * mb.advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb.returns)
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pl += policy_loss.item(); total_vl += value_loss.item()
                total_ent += entropy.item(); n_upd += 1

        return {"policy_loss": total_pl / max(1, n_upd),
                "value_loss": total_vl / max(1, n_upd),
                "entropy": total_ent / max(1, n_upd),
                "mean_reward": batch.rewards.mean().item()}

    def _to_device(self, batch: RolloutBatch, idx) -> RolloutBatch:
        return RolloutBatch(
            boards=batch.boards[idx].to(self.device),
            globals=batch.globals[idx].to(self.device),
            actions=batch.actions[idx].to(self.device),
            log_probs=batch.log_probs[idx].to(self.device),
            advantages=batch.advantages[idx].to(self.device),
            returns=batch.returns[idx].to(self.device),
            action_masks=batch.action_masks[idx].to(self.device),
        )

    # ========== Observation / Action ==========

    def _obs_to_tensors(self, obs: Observation) -> Tuple:
        vec = self.env.to_feature_vector(obs, obs.player_id)
        board_flat = vec[:1575].reshape(7, 15, 15)
        global_feat = vec[1575:1580]
        board_t = torch.from_numpy(board_flat).float()
        glob_t = torch.from_numpy(global_feat).float()
        mask = self._build_action_mask()
        return board_t, glob_t, mask

    def _build_action_mask(self) -> Optional["torch.Tensor"]:
        if not hasattr(self.env, "get_legal_actions"):
            return None
        legal = self.env.get_legal_actions()
        mask = torch.zeros(self.num_actions)
        for a in legal[:500]:
            aid = self._game_action_to_id(a)
            if 0 <= aid < self.num_actions:
                mask[aid] = 1.0
        mask[0] = 1.0  # always allow end turn
        return mask

    def _id_to_game_action(self, aid: int) -> Any:
        if aid == 0: return [[8]]
        a = aid - 1
        if a < 900:
            r, rem = divmod(a, 60)
            c, d = divmod(rem, 4)
            return [[1, r, c, d + 1, 1]]
        return [[8]]

    def _game_action_to_id(self, action: Any) -> int:
        if not isinstance(action, list) or not action: return 0
        cmd = action[0]
        if isinstance(cmd, list) and cmd:
            if cmd[0] == 8: return 0
            if cmd[0] == 1 and len(cmd) >= 5:
                return 1 + cmd[1] * 60 + cmd[2] * 4 + (cmd[3] - 1)
        return 0

    # ========== Evaluation ==========

    def _evaluate(self) -> float:
        if self.eval_opponent:
            from agentbench_frame.arena.match import Match
            from agentbench_frame.agent.rl_agent import RLAgent
            rl_agent = RLAgent(name="Eval", policy_network=self.policy, env=self.env)
            match = Match(self.env, rl_agent, self.eval_opponent)
            return match.run(n_games=self.config.eval_episodes).win_rate

        total_r = 0.0
        for _ in range(self.config.eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                bt, gt, mt = self._obs_to_tensors(obs)
                aid, _, _ = self.policy.get_action(
                    bt.unsqueeze(0).to(self.device), gt.unsqueeze(0).to(self.device),
                    mt.unsqueeze(0).to(self.device) if mt is not None else None,
                    deterministic=True)
                obs, r, done, _ = self.env.step(self._id_to_game_action(aid))
                total_r += r
        return total_r / max(1, self.config.eval_episodes)
