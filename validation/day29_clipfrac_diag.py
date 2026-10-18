"""Day 29 diagnostic: why is the clip fraction 0.3-0.4 in train_minimal.py?

Hypothesis: the actor and critic share a backbone, the value loss has weight c1=1.0 on unnormalised
CartPole returns (targets up to ~100), so value-loss gradients move the shared features and drag the
policy logits along. PPO's clip only removes the *policy* term's incentive; it does nothing about
drift caused by the value term. The paper (Sec 6.1) did not share parameters on MuJoCo.

Method: from identical starting weights and the SAME rollout (T=2048, after a few warm-up
iterations), run the repo's update (K=10 epochs, M=64, lr=3e-4, grad-clip 0.5) under variants and
report the mean clip fraction (as train_minimal.py logs it) and the final approx KL(old||new).

Run:
    OMP_NUM_THREADS=1 python validation/day29_clipfrac_diag.py [warmup_iters]
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, 'papers', '29_PPO'))
from implementation import PPOAgent, ppo_loss  # noqa: E402


class SeparateCriticAC(nn.Module):
    """Same actor as the repo's ActorCritic, but the critic gets its own 64-64 tanh MLP."""
    def __init__(self, shared):
        super().__init__()
        self.actor = shared  # reuse the repo network for the policy (its critic head is ignored)
        self.critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def evaluate(self, obs, actions):
        log_probs, _, entropy = self.actor.evaluate(obs, actions)
        return log_probs, self.critic(obs).squeeze(-1), entropy


def run_update(agent, net, c1, c2, use_policy=True, epochs=10, lr=3e-4):
    net = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    buf = agent.buffer
    obs_all = torch.tensor(buf.obs); act_all = torch.tensor(buf.actions)
    old_lp_all = torch.tensor(buf.log_probs)
    clip_fracs = []
    torch.manual_seed(1); np.random.seed(1)
    for _ in range(epochs):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buf.get_minibatches(64, torch.device('cpu')):
            lp, v, ent = net.evaluate(obs_b, act_b)
            loss, info = ppo_loss(lp, old_lp_b, adv_b, v, ret_b, ent, epsilon=0.2, c1=c1, c2=c2)
            if not use_policy:  # value-loss only: isolates drift caused by the critic's gradient
                loss = c1 * torch.nn.functional.mse_loss(v, ret_b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            clip_fracs.append(info['clip_fraction'])
    with torch.no_grad():
        new_lp, _, _ = net.evaluate(obs_all, act_all)
        log_ratio = new_lp - old_lp_all
        approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()  # unbiased k3 estimator
        final_clip = ((log_ratio.exp() - 1).abs() > 0.2).float().mean().item()
    return float(np.mean(clip_fracs)), final_clip, approx_kl


def main():
    torch.manual_seed(0); np.random.seed(0)
    env = gym.make('CartPole-v1'); env.reset(seed=0)
    agent = PPOAgent(4, 2, T=2048, K=10, M=64, lr=3e-4, c1=1.0, c2=0.01)
    warm = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for it in range(warm):  # warm up so returns are non-trivial
        agent.collect_rollout(env); agent.update()
    mean_r = agent.collect_rollout(env)  # the rollout every variant is evaluated on
    ret = agent.buffer.returns
    print(f'rollout: mean episode reward {mean_r:.1f}; value targets mean {ret.mean():.1f}, max {ret.max():.1f}')
    shared = agent.network
    rows = [
        ('repo default: shared backbone, c1=1.0, c2=0.01', shared, 1.0, 0.01, True),
        ('shared backbone, c1=0.5 (CleanRL/SB3 vf_coef)', shared, 0.5, 0.01, True),
        ('shared backbone, value loss only (no policy term)', shared, 1.0, 0.0, False),
        ('shared backbone, c1=0 (policy + entropy only)', shared, 0.0, 0.01, True),
        ('separate critic MLP (paper Sec 6.1 / CleanRL), c1=1.0', SeparateCriticAC(shared), 1.0, 0.01, True),
    ]
    print(f"{'variant':58s} {'mean clip frac':>15s} {'final clip frac':>16s} {'approx KL':>10s}")
    for name, net, c1, c2, use_pol in rows:
        m, f, kl = run_update(agent, net, c1, c2, use_pol)
        print(f'{name:58s} {m:15.3f} {f:16.3f} {kl:10.4f}')


if __name__ == '__main__':
    main()
