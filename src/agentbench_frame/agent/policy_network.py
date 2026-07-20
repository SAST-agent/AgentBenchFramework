"""
Real neural network policies for RL agents.

Provides PyTorch-based policy networks with:
- CNN for spatial board features
- MLP for global features
- Actor-Critic architecture (shared backbone + separate heads)
- Action masking support
- Save/load with config persistence
"""

from typing import Any, Dict, List, Optional, Tuple
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    HAS_TORCH = False


# ---- Stubs for when torch is not available ----

if not HAS_TORCH:
    class _StubModule:
        """Placeholder for nn.Module when torch is not installed."""
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): raise NotImplementedError("Install torch: pip install torch")
        def to(self, *args, **kwargs): return self
        def parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, *args, **kwargs): pass
        def train(self, *args, **kwargs): return self
        def eval(self): return self

    class _StubNN:
        Module = _StubModule
        Conv2d = _StubModule
        Linear = _StubModule
        Sequential = _StubModule
        ReLU = _StubModule

    nn = _StubNN()  # type: ignore
    F = _StubNN()   # type: ignore


# ---- CNN + MLP Backbone ----

class BoardCNN(nn.Module):
    """CNN encoder for the 15x15 board with 7 channels."""

    def __init__(self, in_channels: int = 7, board_size: int = 15,
                 hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        self.board_size = board_size

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)  # (B, hidden * H * W)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Generals.

    Architecture:
        Board (15,15,7) → CNN → 64*15*15
        Global (5,)      → MLP → 32
        Concat → shared MLP → actor head + critic head

    Supports action masking: pass `action_mask` to forward()
    to zero out illegal action logits.
    """

    def __init__(self,
                 board_channels: int = 7,
                 board_size: int = 15,
                 global_dim: int = 5,
                 num_actions: int = 100,
                 cnn_hidden: int = 64,
                 fc_hidden: int = 256):
        super().__init__()

        # Board encoder
        self.cnn = BoardCNN(board_channels, board_size, cnn_hidden)
        cnn_flat = cnn_hidden * board_size * board_size

        # Global encoder
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
        )

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(cnn_flat + 32, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
        )

        # Heads
        self.actor = nn.Linear(fc_hidden, num_actions)
        self.critic = nn.Linear(fc_hidden, 1)

        # Store shapes for serialization
        self.board_channels = board_channels
        self.board_size = board_size
        self.global_dim = global_dim
        self.num_actions = num_actions

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

        # Actor head: smaller init
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        # Critic head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self,
                board: "torch.Tensor",
                global_feat: "torch.Tensor",
                action_mask: Optional["torch.Tensor"] = None
                ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Args:
            board: (B, C, H, W) board features
            global_feat: (B, D) global features
            action_mask: (B, A) optional mask (1=legal, 0=illegal)

        Returns:
            (action_logits, state_value)
        """
        # Encode
        board_enc = self.cnn(board)           # (B, cnn_flat)
        global_enc = self.global_mlp(global_feat)  # (B, 32)
        combined = torch.cat([board_enc, global_enc], dim=1)  # (B, cnn_flat+32)

        # Shared
        shared = self.shared(combined)  # (B, fc_hidden)

        # Heads
        logits = self.actor(shared)     # (B, num_actions)
        value = self.critic(shared)     # (B, 1)

        # Apply action mask
        if action_mask is not None:
            # Set illegal actions to -inf
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value.squeeze(-1)

    def get_action(self,
                   board: "torch.Tensor",
                   global_feat: "torch.Tensor",
                   action_mask: Optional["torch.Tensor"] = None,
                   deterministic: bool = False
                   ) -> Tuple[int, float, float]:
        """
        Sample an action from the policy.

        Returns:
            (action_id, log_prob, value)
        """
        logits, value = self.forward(board, global_feat, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


# ---- Factory ----

def create_generals_policy(num_actions: int = 100,
                           cnn_hidden: int = 64,
                           fc_hidden: int = 256) -> ActorCritic:
    """Create an ActorCritic network configured for Generals."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for neural network policies. "
                          "Install with: pip install torch")
    return ActorCritic(
        board_channels=7,
        board_size=15,
        global_dim=5,
        num_actions=num_actions,
        cnn_hidden=cnn_hidden,
        fc_hidden=fc_hidden,
    )


def save_policy(policy: ActorCritic, path: str):
    """Save policy and its config to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": {
            "board_channels": policy.board_channels,
            "board_size": policy.board_size,
            "global_dim": policy.global_dim,
            "num_actions": policy.num_actions,
            "cnn_hidden": policy.cnn.conv1.out_channels,
            "fc_hidden": policy.shared[0].out_features,
        },
    }, path)


def load_policy(path: str) -> ActorCritic:
    """Load a saved policy with config."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    policy = ActorCritic(**config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy
