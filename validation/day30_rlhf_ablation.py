"""Day 30 ablation: why the shipped RLHF demo did not learn, and which fix matters.

Self-contained re-implementation of the repo's two buggy pieces next to their fixes, using the repo's
networks (RewardModel, PPOAgent, RLHF_Trainer data collection) so it still runs after the patch:

  reward model training   'orig': pad to max len of segment 1 only (segment 2 truncated), zero padding
                                  counted in the reward sum, ties labelled 'segment 2 better',
                                  no training until the buffer holds 32 pairs
                          'fix' : common padding length, masked sums, ties = 0.5 (paper Sec 2.2.2)
  policy GAE              'orig': no episode-boundary mask, last value 0
                          'fix' : delta and the GAE recursion masked at episode ends, bootstrap V(s_T)
  reward fed to PPO       'learned' r_hat(s) or 'true' (+1 per step: isolates the GAE bug)

Same seeds for every variant; true reward = mean return of 10 evaluation episodes (seeded).
Usage:  python validation/day30_rlhf_ablation.py [cycles=30] [seed=0]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, 'papers', '30_RLHF'))
import implementation as I  # noqa: E402


def collect_pairs(tr, n_pairs, ties):
    """Repo's segment collection; relabel with the original (ties -> 1) or fixed (ties -> 0.5) oracle."""
    n0 = len(tr.preference_buffer)
    tr.collect_and_label_data(n_pairs)
    new = []
    for a, b, _ in tr.preference_buffer[n0:]:
        la, lb = len(a), len(b)  # CartPole: true segment return = number of steps
        lab = (0.5 if la == lb else (0 if la > lb else 1)) if ties == 'fix' else (0 if la > lb else 1)
        new.append((a, b, lab))
    tr.preference_buffer = tr.preference_buffer[:n0] + new


def train_rm(tr, mode, epochs=5, batch_size=32):
    buf = tr.preference_buffer
    if mode == 'orig' and len(buf) < batch_size:
        return float('nan')
    bs = min(batch_size, len(buf))
    idx = np.arange(len(buf)); accs = []
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx) - bs + 1, bs):
            batch = [buf[j] for j in idx[i:i + bs]]
            L = max(len(b[0]) for b in batch) if mode == 'orig' else max(max(len(b[0]), len(b[1])) for b in batch)
            def pad(a):
                out = np.zeros((L, 4), np.float32); n = min(len(a), L); out[:n] = a[:n]
                m = np.zeros(L, np.float32); m[:n] = 1.0
                return out, m
            p1 = [pad(b[0]) for b in batch]; p2 = [pad(b[1]) for b in batch]
            s1 = torch.tensor(np.stack([p[0] for p in p1])); s2 = torch.tensor(np.stack([p[0] for p in p2]))
            r1 = tr.reward_model(s1).squeeze(-1); r2 = tr.reward_model(s2).squeeze(-1)
            if mode == 'fix':
                r1 = r1 * torch.tensor(np.stack([p[1] for p in p1])); r2 = r2 * torch.tensor(np.stack([p[1] for p in p2]))
            logits = torch.stack([r1.sum(1), r2.sum(1)], 1)
            y = torch.tensor([float(b[2]) for b in batch])
            loss = F.cross_entropy(logits, torch.stack([1 - y, y], 1))
            tr.rm_opt.zero_grad(); loss.backward(); tr.rm_opt.step()
            keep = y != 0.5
            if keep.any():
                accs.append((logits.argmax(1).float() == y)[keep].float().mean().item())
    return float(np.mean(accs)) if accs else float('nan')


def policy_step(tr, gae_mode, reward, steps=1000, gamma=0.99, lam=0.95):
    obs, _ = tr.env.reset()
    O, A, LP, V, R, D = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(steps):
            o = torch.tensor(obs, dtype=torch.float32)
            a, lp, _, v = tr.policy.get_action_and_value(o)
            nxt, r_true, term, trunc, _ = tr.env.step(a.item())
            O.append(o); A.append(a); LP.append(lp); V.append(v.item())
            R.append(r_true if reward == 'true' else tr.reward_model(o).item()); D.append(float(term or trunc))
            obs = nxt
            if term or trunc:
                obs, _ = tr.env.reset()
        last = tr.policy.critic(torch.tensor(obs, dtype=torch.float32)).item() if gae_mode == 'fix' else 0.0
    vals = V + [last]; adv = np.zeros(steps); g = 0.0
    for t in reversed(range(steps)):
        nt = 1.0 - D[t] if gae_mode == 'fix' else 1.0
        delta = R[t] + gamma * vals[t + 1] * nt - vals[t]
        g = delta + gamma * lam * nt * g; adv[t] = g
    ret = torch.tensor(adv + np.array(V), dtype=torch.float32)
    advt = torch.tensor(adv, dtype=torch.float32); advt = (advt - advt.mean()) / (advt.std() + 1e-8)
    bo, ba, blp = torch.stack(O), torch.stack(A), torch.stack(LP)
    for _ in range(4):  # same update as the repo: 4 full-batch epochs, clip 0.2, vf 0.5, ent 0.01
        _, nlp, ent, nv = tr.policy.get_action_and_value(bo, ba)
        ratio = torch.exp(nlp - blp)
        pl = -torch.min(ratio * advt, torch.clamp(ratio, 0.8, 1.2) * advt).mean()
        loss = pl + 0.5 * F.mse_loss(nv.squeeze(-1), ret) - 0.01 * ent.mean()
        tr.policy_opt.zero_grad(); loss.backward(); tr.policy_opt.step()


def evaluate(tr, env, n=10):
    out = []
    for k in range(n):
        obs, _ = env.reset(seed=10_000 + k); total = 0.0
        while True:
            with torch.no_grad():
                a, _, _, _ = tr.policy.get_action_and_value(torch.tensor(obs, dtype=torch.float32))
            obs, r, term, trunc, _ = env.step(a.item()); total += r
            if term or trunc:
                break
        out.append(total)
    return float(np.mean(out))


def run(rm_mode, gae_mode, reward, cycles, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make('CartPole-v1'); env.reset(seed=seed); ev = gym.make('CartPole-v1')
    tr = I.RLHF_Trainer(env, 4, 2, segment_length=50)
    hist, acc = [], float('nan')
    for _ in range(cycles):
        if reward == 'learned':
            collect_pairs(tr, 10, ties=rm_mode)
            acc = train_rm(tr, rm_mode)
        policy_step(tr, gae_mode, reward)
        hist.append(evaluate(tr, ev))
    return np.mean(hist[:3]), np.mean(hist[-5:]), acc


if __name__ == '__main__':
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f'uniform-random CartPole policy: ~22 (mean of 200 episodes = 21.8). cycles={cycles}, seed={seed}')
    print(f"{'reward':8s} {'RM training':12s} {'GAE':5s} {'first-3':>8s} {'last-5':>8s} {'final RM acc':>13s} {'time':>6s}")
    for reward, rm_mode, gae_mode in [('learned', 'orig', 'orig'), ('learned', 'orig', 'fix'), ('learned', 'fix', 'orig'),
                                      ('learned', 'fix', 'fix'), ('true', '-', 'orig'), ('true', '-', 'fix')]:
        t0 = time.time()
        f3, l5, acc = run(rm_mode, gae_mode, reward, cycles, seed)
        print(f'{reward:8s} {rm_mode:12s} {gae_mode:5s} {f3:8.1f} {l5:8.1f} {acc:13.2f} {time.time()-t0:5.0f}s', flush=True)
