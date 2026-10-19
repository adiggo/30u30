"""Day 30 (Christiano et al. 2017, RLHF) reference checks.

1. Bradley-Terry: compute_preference_loss == -log P_hat with
   P_hat[s1 > s2] = exp(sum r1) / (exp(sum r1) + exp(sum r2))   (Eq. 1)       -> max |d|
2. Variable-length segments: the masked loss on zero-padded batches equals the loss computed on each
   unpadded pair separately; the unmasked loss (repo before the fix) does not.
3. Ties: label 0.5 gives -0.5 log P - 0.5 log(1-P) (paper Sec 2.2.2: mu uniform); the oracle returns
   0.5 for equal true returns, 0/1 otherwise.
4. Reward normalisation (exercise solution_03): the batch-merge running mean/var equals np.mean /
   np.var of everything seen so far.
5. GAE inside RLHF_Trainer.train_policy_ppo (captured from the function's locals) == explicit
   sum_{l} (gamma*lambda)^l delta_{t+l} truncated at episode ends, with the critic's bootstrap value.
6. Exercises 01-04: solutions satisfy the exercise checks.

Run:  OMP_NUM_THREADS=1 python validation/day30_preference_and_gae.py
"""
import os
import sys

import numpy as np
import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '30_RLHF')
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402

torch.manual_seed(0)


def bt_closed_form(s1, s2):
    return np.exp(s1) / (np.exp(s1) + np.exp(s2))


def check_bt():
    rm = I.RewardModel(4)
    seg1, seg2 = torch.randn(64, 7, 4), torch.randn(64, 7, 4)
    lab = torch.randint(0, 2, (64,))
    loss, _ = I.compute_preference_loss(rm, seg1, seg2, lab)
    with torch.no_grad():
        s1 = rm(seg1).sum(dim=(1, 2)).double().numpy(); s2 = rm(seg2).sum(dim=(1, 2)).double().numpy()
    p = bt_closed_form(s1, s2)
    ref = np.mean(np.where(lab.numpy() == 0, -np.log(p), -np.log(1 - p)))
    d = abs(loss.item() - ref)
    print(f'[1] preference loss vs closed-form Bradley-Terry NLL: |d| = {d:.2e}')
    return d


def check_masking():
    rm = I.RewardModel(4)
    lens1, lens2 = [3, 7, 5, 2], [6, 2, 7, 4]
    raw1 = [torch.randn(n, 4) for n in lens1]; raw2 = [torch.randn(n, 4) for n in lens2]
    lab = torch.tensor([1.0, 0.0, 1.0, 0.0])
    L = max(lens1 + lens2)
    def pad(x):
        out = torch.zeros(L, 4); out[:len(x)] = x
        m = torch.zeros(L); m[:len(x)] = 1
        return out, m
    s1, m1 = zip(*[pad(x) for x in raw1]); s2, m2 = zip(*[pad(x) for x in raw2])
    s1, m1, s2, m2 = torch.stack(s1), torch.stack(m1), torch.stack(s2), torch.stack(m2)
    masked, _ = I.compute_preference_loss(rm, s1, s2, lab, m1, m2)
    unmasked, _ = I.compute_preference_loss(rm, s1, s2, lab)
    with torch.no_grad():
        per = []
        for a, b, y in zip(raw1, raw2, lab):
            p = bt_closed_form(rm(a).sum().item(), rm(b).sum().item())
            per.append(-np.log(p) if y == 0 else -np.log(1 - p))
    ref = float(np.mean(per))
    print(f'[2] masked padded loss vs per-pair unpadded loss: |d| = {abs(masked.item()-ref):.2e}; '
          f'unmasked (pre-fix) |d| = {abs(unmasked.item()-ref):.3f}')
    return abs(masked.item() - ref)


def check_ties_and_oracle():
    rm = I.RewardModel(4)
    seg1, seg2 = torch.randn(5, 4, 4), torch.randn(5, 4, 4)
    loss, acc = I.compute_preference_loss(rm, seg1, seg2, torch.full((5,), 0.5))
    with torch.no_grad():
        p = bt_closed_form(rm(seg1).sum(dim=(1, 2)).double().numpy(), rm(seg2).sum(dim=(1, 2)).double().numpy())
    ref = np.mean(-0.5 * np.log(p) - 0.5 * np.log(1 - p))
    o = I.SyntheticOracle(None)
    answers = (o.query([1, 1, 1], [0, 0, 0]), o.query([0, 0], [1, 1]), o.query([1, 1], [1, 1]))
    print(f'[3] tie label 0.5 vs -0.5 log P - 0.5 log(1-P): |d| = {abs(loss.item()-ref):.2e} (accuracy over decided pairs: {acc}); '
          f'oracle answers (better first, better second, tie) = {answers}')
    assert answers == (0, 1, 0.5)
    return abs(loss.item() - ref)


def check_normalizer():
    sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
    from solution_03_reward_norm import RewardNormalizer
    rng = np.random.default_rng(0)
    rn = RewardNormalizer(); seen = []
    worst = 0.0
    for _ in range(20):
        b = rng.normal(3.0, 2.0, size=int(rng.integers(1, 50)))
        rn.update(b); seen.extend(b)
        worst = max(worst, abs(rn.mean - np.mean(seen)), abs(rn.var - np.var(seen)))
    z = rn.normalize(np.array(seen))
    print(f'[4] running mean/var (batch merge) vs np.mean/np.var over all data: max |d| = {worst:.2e}; '
          f'normalised data mean {z.mean():.1e}, std {z.std():.4f}')
    return worst


def explicit_gae(r, v, d, gamma, lam):
    T = len(r); adv = np.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            total += coef * (r[k] + gamma * v[k + 1] * (1 - d[k]) - v[k])
            if d[k]:
                break
            coef *= gamma * lam
        adv[t] = total
    return adv


def check_policy_gae():
    env = gym.make('CartPole-v1'); env.reset(seed=0)
    tr = I.RLHF_Trainer(env, 4, 2, segment_length=50)
    captured = {}
    def prof(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == 'train_policy_ppo':
            captured.update({k: frame.f_locals[k] for k in ('advs', 'rollout_learned_rewards', 'rollout_vals', 'rollout_dones')})
    sys.setprofile(prof)
    try:
        tr.train_policy_ppo(steps=400)
    finally:
        sys.setprofile(None)
    r = np.array(captured['rollout_learned_rewards']); v = np.array(captured['rollout_vals'])
    d = np.array(captured['rollout_dones'])
    ref = explicit_gae(r, v, d, 0.99, 0.95)
    diff = np.abs(captured['advs'] - ref).max()
    print(f'[5] train_policy_ppo GAE vs explicit truncated sum ({int(d.sum())} episode ends in 400 steps): max |dA| = {diff:.2e}')
    return diff


def check_exercises():
    import importlib
    sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
    s1 = importlib.import_module('solution_01_bradley_terry')
    s2 = importlib.import_module('solution_02_preference_loss')
    s4 = importlib.import_module('solution_04_synthetic_oracle')
    p = s1.compute_preference_probability(torch.tensor(1.0), torch.tensor(0.0)).item()
    l = s2.preference_loss(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0, 1])).item()
    c = s4.SyntheticOracle().query([1, 1, 1], [0, 0, 0])
    print(f'[6] solution_01 P = {p:.4f} (expect {np.e/(np.e+1):.4f}); solution_02 loss = {l:.4f} '
          f'(expect {-np.log(np.e/(np.e+1)):.4f}, exercise threshold < 0.4); solution_04 choice = {c} (expect 0)')


if __name__ == '__main__':
    a = check_bt(); b = check_masking(); c = check_ties_and_oracle(); d = check_normalizer(); e = check_policy_gae()
    check_exercises()
    print(f'SUMMARY day30: BT {a:.1e}, masked {b:.1e}, ties {c:.1e}, normaliser {d:.1e}, GAE {e:.1e}')
