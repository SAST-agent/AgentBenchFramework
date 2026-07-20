"""
Reinforcement Learning Agent

Wraps a neural network policy as a framework-compatible agent.
Supports SB3 and Tianshou policy formats, as well as custom PyTorch models.

The RL agent can:
- Load trained policies from disk
- Act deterministically or stochastically
- Handle both vector and structured observations
- Support action masking for legal move filtering
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from agentbench_frame.agent.base import BaseAgent


class PolicyNetwork:
    """
    A simple feedforward policy network for game AI.

    Supports:
    - Multi-input: board features (CNN) + global features (MLP)
    - Action heads: discrete action selection
    - Action masking for legal moves
    """

    def __init__(self,
                 board_shape: Tuple[int, int, int] = (15, 15, 7),
                 global_dim: int = 5,
                 num_actions: int = 100,
                 hidden_dim: int = 256):
        self.board_shape = board_shape
        self.global_dim = global_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

    def predict(self, observation: Any) -> Tuple[int, Any]:
        """
        Predict action from observation vector.

        Args:
            observation: Flat feature vector (list, array, or similar)

        Returns:
            (action_id, action_probabilities)
        """
        # Placeholder: random policy (works with or without numpy)
        if HAS_NUMPY:
            probs = np.ones(self.num_actions) / self.num_actions
            action = np.random.choice(self.num_actions)
        else:
            probs = [1.0 / self.num_actions] * self.num_actions
            action = random.randint(0, self.num_actions - 1)
        return action, probs

    def save(self, path: str):
        """Save model weights."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "board_shape": self.board_shape,
                "global_dim": self.global_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PolicyNetwork":
        """Load model weights."""
        import pickle
        with open(path, "rb") as f:
            config = pickle.load(f)
        return cls(**config)


class RLAgent(BaseAgent):
    """
    An agent backed by a learned policy (neural network).

    Can be used with:
    - Stable-Baselines3 trained models
    - Tianshou trained policies
    - Custom PolicyNetwork instances
    - ONNX exported models

    The agent handles the observation-to-vector conversion and
    action selection from the policy output.
    """

    def __init__(self,
                 name: str = "RLAgent",
                 policy: Optional[PolicyNetwork] = None,
                 policy_network: Any = None,   # ActorCritic (real torch policy)
                 deterministic: bool = True,
                 env: Any = None):
        """
        Args:
            name: Agent identifier
            policy: Legacy PolicyNetwork (placeholder, no training)
            policy_network: Real ActorCritic torch policy (from PPOTrainer)
            deterministic: If True, always pick the best action
            env: Reference to the environment (for action space info)
        """
        super().__init__(name=name)
        self.policy = policy or PolicyNetwork()
        self._policy_network = policy_network  # real torch policy
        self.deterministic = deterministic
        self.env = env
        self._sb3_model = None
        self._tianshou_policy = None

    def load_sb3(self, model_path: str):
        """Load a Stable-Baselines3 trained model."""
        try:
            from stable_baselines3 import PPO
            self._sb3_model = PPO.load(model_path)
        except ImportError:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

    def load_tianshou(self, policy_path: str):
        """Load a Tianshou trained policy."""
        try:
            import torch
            self._tianshou_policy = torch.load(policy_path, weights_only=False)
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

    def act(self, observation: Any) -> Any:
        """
        Choose an action using the learned policy.

        Backends (checked in order):
        1. Real ActorCritic torch policy (from PPOTrainer)
        2. SB3 model (if loaded)
        3. Tianshou policy (if loaded)
        4. Legacy PolicyNetwork (random)
        """
        # 1. Real torch policy
        if self._policy_network is not None and HAS_TORCH and HAS_NUMPY:
            return self._act_with_torch_policy(observation)

        # 2. SB3 model
        if self._sb3_model is not None:
            if HAS_NUMPY and isinstance(observation, np.ndarray):
                obs_vec = observation
            elif hasattr(observation, "state"):
                obs_vec = self._obs_to_vector(observation)
            else:
                obs_vec = self._dict_to_list(observation)

            action, _states = self._sb3_model.predict(
                obs_vec, deterministic=self.deterministic
            )
            return action

        # Try Tianshou
        if self._tianshou_policy is not None:
            import torch
            if HAS_NUMPY:
                if isinstance(observation, np.ndarray):
                    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                else:
                    vec = self._obs_to_vector(observation)
                    obs_tensor = torch.from_numpy(vec).float().unsqueeze(0)
            else:
                vec = self._obs_to_vector(observation)
                obs_tensor = torch.tensor(vec).float().unsqueeze(0)

            with torch.no_grad():
                logits = self._tianshou_policy(obs_tensor)
                if self.deterministic:
                    action = logits.argmax(dim=1).item()
                else:
                    probs = torch.softmax(logits, dim=1)
                    action = torch.multinomial(probs, 1).item()
            return action

        # Fall back to default policy
        if hasattr(self.env, "to_feature_vector"):
            obs_vec = self.env.to_feature_vector(observation, observation.player_id)
        elif HAS_NUMPY and isinstance(observation, np.ndarray):
            obs_vec = observation
        else:
            obs_vec = self._dict_to_list(observation)

        action_id, _ = self.policy.predict(obs_vec)

        # Convert action_id to game action
        return self._id_to_action(action_id)

    def _obs_to_vector(self, observation) -> Any:
        """Convert structured observation to vector."""
        if self.env and hasattr(self.env, "to_feature_vector"):
            return self.env.to_feature_vector(observation, observation.player_id)
        if HAS_NUMPY:
            return np.zeros(self.policy.global_dim)
        return [0.0] * self.policy.global_dim

    def _dict_to_list(self, observation) -> Any:
        """Convert observation dict to flat list."""
        if isinstance(observation, dict):
            return list(observation.values())
        if hasattr(observation, "state"):
            return list(observation.state.values())
        return [0.0] * self.policy.global_dim

    def _act_with_torch_policy(self, observation: Any) -> Any:
        """Use the real ActorCritic torch policy for inference."""
        import torch

        if not hasattr(self.env, "to_feature_vector"):
            return [[8]]

        vec = self.env.to_feature_vector(observation, observation.player_id)
        board_flat = vec[:1575].reshape(7, 15, 15)
        global_feat = vec[1575:1580]
        board_t = torch.from_numpy(board_flat).float().unsqueeze(0)
        glob_t = torch.from_numpy(global_feat).float().unsqueeze(0)

        # Build action mask
        num_actions = self._policy_network.num_actions
        mask_t = torch.ones(1, num_actions)
        if hasattr(self.env, "get_legal_actions"):
            legal = self.env.get_legal_actions()
            mask_t = torch.zeros(1, num_actions)
            for a in legal[:500]:
                if isinstance(a, list) and a:
                    cmd = a[0]
                    if isinstance(cmd, list) and cmd:
                        if cmd[0] == 8: mask_t[0, 0] = 1.0
                        elif cmd[0] == 1 and len(cmd) >= 5:
                            aid = 1 + cmd[1] * 60 + cmd[2] * 4 + (cmd[3] - 1)
                            if 0 <= aid < num_actions:
                                mask_t[0, aid] = 1.0
            mask_t[0, 0] = 1.0  # always allow end turn

        with torch.no_grad():
            aid, _, _ = self._policy_network.get_action(
                board_t, glob_t, mask_t, deterministic=self.deterministic)

        # action_id → game action
        if aid == 0: return [[8]]
        a = aid - 1
        if a < 900:
            r, rem = divmod(a, 60)
            c, d = divmod(rem, 4)
            return [[1, r, c, d + 1, 1]]
        return [[8]]

    def _id_to_action(self, action_id: int) -> Any:
        """Convert action ID to environment-compatible action."""
        # Default: just end turn
        return [[8]]

    def reset(self):
        """Reset agent state."""
        pass

    def save(self, path: str):
        """Save the policy."""
        self.policy.save(path)

    @classmethod
    def load(cls, path: str) -> "RLAgent":
        """Load a saved RL agent."""
        agent = cls()
        agent.policy = PolicyNetwork.load(path)
        return agent
