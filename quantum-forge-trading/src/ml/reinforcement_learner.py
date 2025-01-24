import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
import cupy as cp  # Added for GPU acceleration
from typing import Tuple, List

class TradePolicyGradient:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Enhanced model architecture with residual connections
        self.model = nn.ModuleDict({
            'feature_extractor': nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'policy_head': nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'value_head': nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        }).to(self.device)
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        # Added learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000
        )
        
        # Added experience replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(max_size=10000)
        
    @torch.no_grad()
    def update_model(self, trade_outcome: dict) -> float:
        """Optimized quantum-inspired policy gradient update"""
        self.replay_buffer.add(trade_outcome)
        
        if len(self.replay_buffer) < 64:  # Minimum batch size
            return 0.0
            
        batch = self.replay_buffer.sample(64)
        
        with autocast():
            states = torch.tensor(batch['states'], device=self.device)
            features = self.model['feature_extractor'](states)
            
            policy_loss = self._compute_policy_loss(features, batch)
            value_loss = self._compute_value_loss(features, batch)
            
            total_loss = policy_loss + 0.5 * value_loss
            
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return total_loss.item()
        
    @torch.jit.script
    def predict_action(self, market_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Optimized hybrid classical-quantum inference"""
        market_state = market_state.to(self.device)
        
        with autocast():
            features = self.model['feature_extractor'](market_state)
            policy = self.model['policy_head'](features)
            value = self.model['value_head'](features)
            
        return F.softmax(policy, dim=-1), value.item()
        
    def _compute_policy_loss(self, features: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute policy gradient loss with importance sampling"""
        policy = self.model['policy_head'](features)
        advantages = torch.tensor(batch['advantages'], device=self.device)
        old_policy = torch.tensor(batch['old_policy'], device=self.device)
        
        ratio = torch.exp(policy - old_policy)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        
        return -torch.min(
            ratio * advantages,
            clip_ratio * advantages
        ).mean()
        
    def _compute_value_loss(self, features: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute value function loss"""
        value = self.model['value_head'](features)
        returns = torch.tensor(batch['returns'], device=self.device)
        
        return F.mse_loss(value, returns)

class PrioritizedReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        
    def add(self, experience: dict) -> None:
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max(self.priorities.max(), 1.0)
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size: int) -> dict:
        probs = self.priorities[:len(self.buffer)] ** 0.6
        probs /= probs.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probs
        )
        
        samples = [self.buffer[idx] for idx in indices]
        return {
            'states': np.stack([s['state'] for s in samples]),
            'actions': np.stack([s['action'] for s in samples]),
            'rewards': np.stack([s['reward'] for s in samples]),
            'next_states': np.stack([s['next_state'] for s in samples]),
            'dones': np.stack([s['done'] for s in samples])
        }
