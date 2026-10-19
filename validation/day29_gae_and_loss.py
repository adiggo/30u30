"""Day 29 (PPO) reference checks.

1. GAE: RolloutBuffer.compute_advantages (backward recursion with done masks) vs an independent
   explicit sum  A_t = sum_{l>=0} (gamma*lambda)^l delta_{t+l}, truncated at the first episode end,
   with delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t)   (PPO paper Eqs. 11-12; GAE paper).
   Also: returns == advantages + values, and lambda=1 / lambda=0 special cases.
2. Clipped surrogate (Eq. 7) in ppo_loss vs a hand computation, including both clip regimes, and
   the gradient (zero exactly where the clipped branch is active and selected by the min).
3. Value term == mean squared error, entropy == closed-form Bernoulli/categorical entropy, and the
   total loss == policy + c1*value - c2*entropy (Eq. 9 sign convention).
4. Solutions 01/02 agree with the exercise tests (monkey-patched).

Run:  OMP_NUM_THREADS=1 python validation/day29_gae_and_loss.py
"""
import importlib
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '29_PPO')
sys.path.insert(0, DAY)
from implementation import RolloutBuffer, ppo_loss, ActorCritic  # noqa: E402


def explicit_gae(r, v, d, gamma, lam):
    """Independent implementation: explicit forward sum, stopping after the first terminal step."""
    T = len(r)
    adv = np.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            delta = r[k] + gamma * v[k + 1] * (1.0 - d[k]) - v[k]
            total += coef * delta
            if d[k]:            # episode ended at k: nothing after k belongs to this episode
                break
            coef *= gamma * lam
        adv[t] = total
    return adv


def check_gae():
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(200):
        T = int(rng.integers(5, 60))
        gamma, lam = float(rng.uniform(0.9, 1.0)), float(rng.uniform(0.0, 1.0))
        r = rng.normal(size=T).astype(np.float32)
        v = rng.normal(size=T + 1).astype(np.float32)
        d = (rng.random(T) < 0.15).astype(np.float32)
        buf = RolloutBuffer(T, obs_dim=3, act_dim=1, gamma=gamma, lambda_=lam)
        for t in range(T):
            buf.store(np.zeros(3), 0, r[t], d[t], 0.0, v[t])
        buf.compute_advantages(last_value=float(v[T]))
        ref = explicit_gae(r.astype(np.float64), v.astype(np.float64), d, gamma, lam)
        worst = max(worst, np.max(np.abs(buf.advantages - ref)))
        assert np.allclose(buf.returns, buf.advantages + v[:T], atol=1e-6)
    print(f'[GAE] backward recursion vs explicit sum, 200 random rollouts with episode ends: max |dA| = {worst:.2e}')
    assert worst < 1e-4
    # lambda = 1, gamma = 1, zero values: advantage = undiscounted reward-to-go within the episode
    r = np.array([1, 1, 1, 1, 1], np.float32); d = np.array([0, 1, 0, 0, 0], np.float32)
    buf = RolloutBuffer(5, 1, 1, gamma=1.0, lambda_=1.0)
    for t in range(5):
        buf.store(np.zeros(1), 0, r[t], d[t], 0.0, 0.0)
    buf.compute_advantages(0.0)
    print(f'[GAE] lambda=1, gamma=1, V=0, done at t=1: A = {buf.advantages.tolist()} (expect [2, 1, 3, 2, 1])')
    assert np.allclose(buf.advantages, [2, 1, 3, 2, 1])
    return worst


def check_clip_loss():
    eps = 0.2
    lp_old = torch.zeros(6)
    # ratios 1.5 (A>0, clipped), 0.5 (A<0, clipped), 1.1 (A>0, inside), 0.5 (A>0, unclipped: min picks r*A),
    # 1.5 (A<0, unclipped: min picks r*A), 1.0 (A>0)
    ratios = torch.tensor([1.5, 0.5, 1.1, 0.5, 1.5, 1.0])
    adv = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 2.0])
    lp_new = torch.log(ratios).clone().requires_grad_(True)
    # ppo_loss normalises advantages inside; reproduce that by hand
    a = (adv - adv.mean()) / (adv.std() + 1e-8)
    hand = -torch.minimum(ratios * a, torch.clamp(ratios, 1 - eps, 1 + eps) * a).mean()
    values = torch.tensor([0.5, 1.0, 0.0, 2.0, -1.0, 0.3])
    returns = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    probs = torch.tensor([0.3, 0.5, 0.9, 0.1, 0.6, 0.5])
    ent = -(probs * probs.log() + (1 - probs) * (1 - probs).log())  # Bernoulli entropy, closed form
    total, info = ppo_loss(lp_new, lp_old, adv, values, returns, ent, epsilon=eps, c1=0.7, c2=0.05)
    d_pol = abs(info['policy_loss'] - hand.item())
    d_val = abs(info['value_loss'] - ((values - returns) ** 2).mean().item())
    d_ent = abs(info['entropy'] - ent.mean().item())
    d_tot = abs(total.item() - (hand.item() + 0.7 * ((values - returns) ** 2).mean().item() - 0.05 * ent.mean().item()))
    print(f'[CLIP] policy term vs hand computation |d| = {d_pol:.2e}; value MSE |d| = {d_val:.2e}; '
          f'entropy |d| = {d_ent:.2e}; total (Eq. 9 signs) |d| = {d_tot:.2e}')
    # gradient check: only samples where the unclipped branch is selected carry gradient
    total.backward()
    g = lp_new.grad
    selected_unclipped = (ratios * a) <= (torch.clamp(ratios, 1 - eps, 1 + eps) * a)
    clipped_active = (~selected_unclipped) | (((ratios < 1 - eps) | (ratios > 1 + eps)) & (ratios * a == torch.clamp(ratios, 1 - eps, 1 + eps) * a))
    print(f'[CLIP] d(loss)/d(log r) = {[round(x, 4) for x in g.tolist()]}  (zero where clipped branch selected: '
          f'{[bool(x) for x in clipped_active.tolist()]})')
    expected_grad = torch.where(clipped_active, torch.zeros(6), -(ratios * a) / 6)
    d_grad = (g - expected_grad).abs().max().item()
    print(f'[CLIP] gradient vs analytic (-r*A/n on unclipped samples, 0 on clipped): max |d| = {d_grad:.2e}')
    # entropy from the network's Categorical vs closed form
    net = ActorCritic(4, 3)
    obs = torch.randn(5, 4)
    _, _, ent_net = net.evaluate(obs, torch.zeros(5, dtype=torch.long))
    with torch.no_grad():
        p = torch.softmax(net.actor_head(net.backbone(obs)), -1)
    d_cat = (ent_net - (-(p * p.log()).sum(-1))).abs().max().item()
    print(f'[ENT] Categorical entropy vs -sum p log p: max |d| = {d_cat:.2e}')
    return max(d_pol, d_val, d_ent, d_tot, d_grad, d_cat)


def check_exercises():
    sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
    ok = []
    for ex, sol, fn, test in [('exercise_01_clipped_loss', 'solution_01_clipped_loss', 'clipped_surrogate_loss', 'test_clipped_surrogate_loss'),
                              ('exercise_02_gae', 'solution_02_gae', 'compute_gae', 'test_gae'),
                              ('exercise_03_actor_critic', 'solution_03_actor_critic', 'ActorCriticNetwork', 'test_actor_critic')]:
        exm = importlib.import_module(f'exercises.{ex}')
        solm = importlib.import_module(sol)
        setattr(exm, fn, getattr(solm, fn))
        torch.manual_seed(0)
        getattr(exm, test)()
        ok.append(ex)
    # solution_02 GAE vs the explicit sum too
    solm = importlib.import_module('solution_02_gae')
    rng = np.random.default_rng(1); worst = 0.0
    for _ in range(100):
        T = int(rng.integers(5, 40)); r = rng.normal(size=T); v = rng.normal(size=T + 1); d = (rng.random(T) < 0.2).astype(float)
        worst = max(worst, np.abs(solm.compute_gae(r, v, d, 0.99, 0.95) - explicit_gae(r, v, d, 0.99, 0.95)).max())
    print(f'[EX] tests pass with solutions swapped in: {ok}; solution_02 GAE vs explicit sum max |d| = {worst:.2e}')
    return worst


if __name__ == '__main__':
    g = check_gae()
    c = check_clip_loss()
    e = check_exercises()
    print(f'SUMMARY day29: GAE max|d|={g:.2e}, clip/value/entropy max|d|={c:.2e}, exercise GAE max|d|={e:.2e}')
