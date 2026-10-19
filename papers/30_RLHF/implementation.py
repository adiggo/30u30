"""
implementation.py - Core RLHF Modules for Day 30

This file implements the essential components of Reinforcement Learning from Human Feedback:
1.  RewardModel: A neural network that predicts scalar rewards.
2.  PreferenceLoss: The Bradley-Terry cross-entropy loss function.
3.  SyntheticOracle: A simulated human labeler.
4.  PPOAgent: A policy that optimizes against the learned reward.

References:
    1. Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"
       https://arxiv.org/abs/1706.03741
    2. Ouyang et al. (2022) "Training language models to follow instructions with human feedback"
       https://arxiv.org/abs/2203.02155
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

class RewardModel(nn.Module):
    """
    Predicts a scalar reward r(s) for a given observation s.
    In the paper (Section 2), this is the function \\hat{r}.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Outputs scalar reward
        )

    def forward(self, obs):
        return self.net(obs)

def compute_preference_loss(reward_model, segment1, segment2, label, mask1=None, mask2=None):
    """
    Computes the cross-entropy loss for pairwise preferences (Eq 2 in Christiano et al.).
    
    Args:
        reward_model: The RewardModel network.
        segment1: Tensor of shape (batch, len, obs_dim) - the first trajectory segment.
        segment2: Tensor of shape (batch, len, obs_dim) - the second trajectory segment.
        label: Tensor of shape (batch,) - 0 if segment1 is preferred, 1 if segment2 is preferred.
               (Or soft labels, i.e., probability that segment2 is preferred).
        mask1, mask2: Optional (batch, len) tensors, 1 for real timesteps and 0 for padding.
    
    Returns:
        loss: Scalar tensor.
        accuracy: Scalar float.
    """
    # 1. Compute rewards for each timestep in the segments
    # Shape: (batch, len, 1)
    r1 = reward_model(segment1) 
    r2 = reward_model(segment2)
    # [PATCH 2026-10-18] zero-padded timesteps must not add r_hat(0) to the segment sums;
    # without the mask, short (worse) segments get padding reward and the RM cannot separate them.
    if mask1 is not None:
        r1 = r1 * mask1.unsqueeze(-1)
    if mask2 is not None:
        r2 = r2 * mask2.unsqueeze(-1)
    
    # 2. Sum rewards over the segment (Section 2.1)
    # Shape: (batch,)
    r1_sum = torch.sum(r1, dim=1).squeeze(-1)
    r2_sum = torch.sum(r2, dim=1).squeeze(-1)
    
    # 3. Compute preference probability using Bradley-Terry model (Eq 1)
    # P(1 > 2) = exp(r1) / (exp(r1) + exp(r2))
    # P(2 > 1) = exp(r2) / (exp(r1) + exp(r2))
    # Logits for cross_entropy are just the reward sums! 
    # Because softmax([r1_sum, r2_sum]) gives exactly the Bradley-Terry probabilities.
    logits = torch.stack([r1_sum, r2_sum], dim=1) # (batch, 2)

    # [PATCH 2026-10-18] the docstring promised soft labels but cross_entropy(logits, float_label)
    # raises; a float label is P(segment2 preferred), and 0.5 is a tie (mu uniform, Sec 2.2.2).
    if label.is_floating_point():
        loss = F.cross_entropy(logits, torch.stack([1.0 - label, label], dim=1))
        with torch.no_grad():
            decided = label != 0.5
            preds = torch.argmax(logits, dim=1)
            accuracy = ((preds == label.round().long())[decided].float().mean()
                        if decided.any() else torch.tensor(float('nan')))
        return loss, accuracy.item()
    
    # 4. Cross-entropy loss (Eq 2)
    loss = F.cross_entropy(logits, label)
    
    # Calculate accuracy
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == label).float().mean()
        
    return loss, accuracy.item()

class SyntheticOracle:
    """
    Simulates a human evaluator.
    In a real application, this would be replaced by a UI showing clips to a human.
    Here, we use the environment's ground-truth reward to generate labels.
    """
    def __init__(self, env):
        self.env = env # Just for reference, typically hard to simulate step-by-step
        
    def query(self, segment1_rewards, segment2_rewards):
        """
        Returns 0 if segment1 is better, 1 if segment2 is better.
        The paper typically adds noise to simulate human error (Section 3),
        but we'll keep it deterministic for stability in this minimal example.
        """
        sum1 = np.sum(segment1_rewards)
        sum2 = np.sum(segment2_rewards)
        # [PATCH 2026-10-18] ties were labelled "segment 2 better"; the paper's oracle outputs
        # indifference (footnote 4), stored as a uniform label (Sec 2.2.2).
        if sum1 == sum2:
            return 0.5
        return 0 if sum1 > sum2 else 1

class PPOAgent(nn.Module):
    """
    Standard PPO actor-critic, similar to Day 29.
    Included here to make Day 30 self-contained.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)

class RLHF_Trainer:
    """
    Manages the full loop:
    1. Collect trajectories with Policy.
    2. Label pairs with Oracle.
    3. Train Reward Model.
    4. Train Policy with PPO using Learned Reward.
    """
    def __init__(self, env, obs_dim, act_dim, segment_length=50, buffer_capacity=1000):
        self.env = env
        self.segment_length = segment_length
        
        # Models
        self.policy = PPOAgent(obs_dim, act_dim)
        self.reward_model = RewardModel(obs_dim)
        self.oracle = SyntheticOracle(env)
        
        # Optimizers
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.rm_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
        # Buffers
        self.preference_buffer = [] # Stores (seg1, seg2, label)
        self.buffer_capacity = buffer_capacity

    def collect_and_label_data(self, num_new_pairs):
        """Phase 1 & 2: Collect segments and get labels (Section 2.2)."""
        new_segments = []
        
        # Collect segments
        # Note: Inefficient implementation for clarity. Real implementations use parallel storage.
        obs, _ = self.env.reset()
        current_segment_obs = []
        current_segment_true_rewards = []
        
        while len(new_segments) < num_new_pairs * 2: # Need 2x segments for x pairs
            # Step policy
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = self.policy.get_action_and_value(obs_t)
            
            next_obs, true_reward, done, truncated, _ = self.env.step(action.item())
            
            current_segment_obs.append(obs)
            current_segment_true_rewards.append(true_reward)
            obs = next_obs
            
            if len(current_segment_obs) == self.segment_length or done or truncated:
                new_segments.append({
                    'obs': np.array(current_segment_obs),
                    'true_rewards': np.array(current_segment_true_rewards)
                })
                current_segment_obs = []
                current_segment_true_rewards = []
                if done or truncated:
                    # [PATCH 2026-10-17] was `assert env is not None` (NameError)
                    assert self.env is not None
                    obs, _ = self.env.reset()
                    
        # Pair up and label
        for i in range(num_new_pairs):
            seg1 = new_segments[2*i]
            seg2 = new_segments[2*i+1]
            
            label = self.oracle.query(seg1['true_rewards'], seg2['true_rewards'])
            self.preference_buffer.append((seg1['obs'], seg2['obs'], label))
            
        # Keep buffer size limited
        if len(self.preference_buffer) > self.buffer_capacity:
            self.preference_buffer = self.preference_buffer[-self.buffer_capacity:]
            
    def train_reward_model(self, epochs=1, batch_size=32):
        """Phase 3: Train Reward Model on labeled pairs."""
        # [PATCH 2026-10-18] was `return 0.0, 0.0` when the buffer held < batch_size pairs: the RM
        # silently did not train for the first cycles and "0% accuracy" was printed.
        if len(self.preference_buffer) == 0:
            return float('nan'), float('nan')
        batch_size = min(batch_size, len(self.preference_buffer))
            
        total_loss = 0
        total_acc = 0
        updates = 0
        acc_updates = 0  # [PATCH 2026-10-18]
        
        indices = np.arange(len(self.preference_buffer))
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size: continue
                
                batch = [self.preference_buffer[idx] for idx in batch_indices]
                
                # Pad sequences to same length for batching (or just use fixed length)
                # Here we assume fixed length for simplicity or just truncate/pad
                # [PATCH 2026-10-18] was the max over seg1 only, so longer seg2s were truncated
                max_len = max(max(len(b[0]), len(b[1])) for b in batch)
                
                # Helper to pad
                def pad(arr, length):
                    d = arr.shape[1]
                    res = np.zeros((length, d), dtype=np.float32)
                    l = min(len(arr), length)
                    res[:l] = arr[:l]
                    return res

                # [PATCH 2026-10-18] masks mark real timesteps (1) vs padding (0)
                def valid(arr, length):
                    m = np.zeros(length, dtype=np.float32)
                    m[:min(len(arr), length)] = 1.0
                    return m

                s1 = torch.tensor(np.stack([pad(b[0], max_len) for b in batch]), dtype=torch.float32)
                s2 = torch.tensor(np.stack([pad(b[1], max_len) for b in batch]), dtype=torch.float32)
                m1 = torch.tensor(np.stack([valid(b[0], max_len) for b in batch]))
                m2 = torch.tensor(np.stack([valid(b[1], max_len) for b in batch]))
                # [PATCH 2026-10-18] float labels so a tie (0.5) is not truncated to 0
                labels = torch.tensor([float(b[2]) for b in batch], dtype=torch.float32)
                
                loss, acc = compute_preference_loss(self.reward_model, s1, s2, labels, m1, m2)
                
                self.rm_opt.zero_grad()
                loss.backward()
                self.rm_opt.step()
                
                total_loss += loss.item()
                # [PATCH 2026-10-18] accuracy is over decided pairs only; skip all-tie batches
                if not np.isnan(acc):
                    total_acc += acc
                    acc_updates += 1
                updates += 1
                
        return (total_loss / (updates + 1e-8),
                total_acc / acc_updates if acc_updates else float('nan'))

    def train_policy_ppo(self, steps=1000):
        """Phase 4: Optimize Policy using Learned Reward (PPO)."""
        # Collect rollout with learned rewards
        obs, _ = self.env.reset()
        rollout_obs = []
        rollout_acts = []
        rollout_logprobs = []
        rollout_vals = []
        rollout_learned_rewards = []
        rollout_dones = []  # [PATCH 2026-10-18] episode ends, needed by GAE below
        
        # 1. Collect Data using Learned Reward
        for _ in range(steps):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            rollout_obs.append(obs_t)
            
            with torch.no_grad():
                action, logprob, _, val = self.policy.get_action_and_value(obs_t)
                learned_r = self.reward_model(obs_t).item()
                
            rollout_acts.append(action)
            rollout_logprobs.append(logprob)
            rollout_vals.append(val.item())
            rollout_learned_rewards.append(learned_r)
            
            next_obs, _, done, truncated, _ = self.env.step(action.item())
            obs = next_obs
            rollout_dones.append(float(done or truncated))
            
            if done or truncated:
                obs, _ = self.env.reset()
                
        # 2. Compute Advantages (GAE) using Learned Rewards
        # (Simplified GAE: just return - value)
        returns = []
        gae = 0
        gamma = 0.99
        lam = 0.95
        
        # [PATCH 2026-10-18] bootstrap from V(last obs) instead of a dummy 0 (a fake episode end)
        with torch.no_grad():
            rollout_vals.append(self.policy.critic(torch.tensor(obs, dtype=torch.float32)).item())
        advs = np.zeros(steps)
        
        for t in reversed(range(steps)):
            # [PATCH 2026-10-18] mask at episode ends: without it GAE bootstraps through env.reset(),
            # so falling over looks free; in CartPole that removes the only signal (even the true
            # +1 reward then trains the policy to fall faster).
            nonterminal = 1.0 - rollout_dones[t]
            delta = rollout_learned_rewards[t] + gamma * rollout_vals[t+1] * nonterminal - rollout_vals[t]
            gae = delta + gamma * lam * nonterminal * gae
            advs[t] = gae
            
        returns = torch.tensor(advs + rollout_vals[:-1], dtype=torch.float32)
        advantages = torch.tensor(advs, dtype=torch.float32)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. PPO Update (Simplified Epochs)
        b_obs = torch.stack(rollout_obs)
        b_acts = torch.stack(rollout_acts)
        b_logprobs = torch.stack(rollout_logprobs)
        
        clip_param = 0.2
        
        for _ in range(4): # 4 epochs
            _, new_logprobs, entropy, new_vals = self.policy.get_action_and_value(b_obs, b_acts)
            ratio = torch.exp(new_logprobs - b_logprobs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_vals.squeeze(-1), returns)
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()
            
        return np.mean(rollout_learned_rewards)

if __name__ == "__main__":
    print("RLHF Implementation loaded.")
    print("See train_minimal.py to run the full loop.")
