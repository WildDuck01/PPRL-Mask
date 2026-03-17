"""Contextual bandit agent for Route-A (mask selection).

This agent is intentionally simple and stable for semi-supervised segmentation:
- Uses epsilon-greedy exploration over a small discrete action space (K=4).
- Learns to predict immediate reward (gamma=0), i.e., contextual bandit.
- Uses a small replay buffer and MSE regression on rewards.

Note: For strict reproducibility, set seeds outside and pass seed here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RLPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BanditConfig:
    state_dim: int = 258
    action_dim: int = 4
    hidden_dim: int = 128
    lr: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 1e-3
    replay_capacity: int = 10_000


class ContextualBanditAgent:
    """Epsilon-greedy contextual bandit with reward regression."""

    def __init__(self, cfg: BanditConfig, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.cfg = cfg
        self.policy_net = RLPolicyNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).cuda()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

        self.memory: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        self.total_steps = 0

    def get_epsilon(self) -> float:
        c = self.cfg
        eps = c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(-1.0 * self.total_steps * c.epsilon_decay)
        return float(eps)

    def choose_action(self, state: np.ndarray, train: bool = True) -> int:
        """Return action in [0, action_dim)."""
        self.total_steps += 1
        eps = self.get_epsilon() if train else 0.0

        if train and random.random() < eps:
            return random.randrange(self.cfg.action_dim)

        st = torch.tensor(state[None, :], dtype=torch.float32, device='cuda')
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(st)
        return int(torch.argmax(q, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        # Basic validation
        if not (isinstance(state, np.ndarray) and state.ndim == 1 and state.shape[0] == self.cfg.state_dim):
            return
        if not (isinstance(next_state, np.ndarray) and next_state.ndim == 1 and next_state.shape[0] == self.cfg.state_dim):
            return
        self.memory.append((state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32)))
        if len(self.memory) > self.cfg.replay_capacity:
            self.memory.pop(0)

    def update(self, batch_size: int = 32) -> Optional[float]:
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device='cuda')
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64, device='cuda').unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device='cuda').unsqueeze(1)

        self.policy_net.train()
        pred = self.policy_net(s).gather(1, a)
        # gamma=0 => target is immediate reward
        loss = F.mse_loss(pred, r)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


def extract_state_from_encoder_feature(
    feature_map: torch.Tensor,
    entropy_mean: float,
    disagreement_mean: float
        ) -> np.ndarray:
    """
    Args:
        feature_map: encoder feature [B,C,h,w]
        entropy_mean: teacher prediction entropy (uncertainty)
        disagreement_mean: KL(pT || pS) (teacher-student disagreement)
    """
    with torch.no_grad():
        pooled = F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)
        pooled = pooled.mean(dim=0, keepdim=True)  # [1,C]

        metrics = torch.tensor(
            [[entropy_mean, disagreement_mean]],
            dtype=pooled.dtype,
            device=pooled.device
        )

        state = torch.cat([pooled, metrics], dim=1)
    return state.squeeze(0).detach().cpu().numpy()



def extract_state_from_feature_3d(feature_map: torch.Tensor,
                                  entropy_mean: float,
                                  disagreement_mean: float) -> np.ndarray:
    """
    feature_map: [B,C,D,H,W] 或 [B,C,H,W]
    return: np.ndarray [state_dim]
    """
    with torch.no_grad():
        pooled = F.adaptive_avg_pool3d(feature_map, 1).view(feature_map.size(0), -1)

        pooled = pooled.mean(dim=0, keepdim=True)  # [1,C]
        metrics = torch.tensor([[entropy_mean, disagreement_mean]],
                               dtype=pooled.dtype, device=pooled.device)  # [1,2]
        state = torch.cat([pooled, metrics], dim=1)  # [1,C+2]
        return state.squeeze(0).detach().cpu().numpy().astype(np.float32)


