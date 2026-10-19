"""Day 30 diagnostic harness (first-pass diagnosis; see day30_rlhf_ablation.py for the clean ablation).
Flags are '+'-joined: gae (done-masked GAE), rmfix (masked, common-length RM training), tiesdrop / tieshalf,
norm (zero-mean/unit-std learned reward per rollout). Without rmfix it calls RLHF_Trainer.train_reward_model,
which is now the PATCHED version. Example: python validation/day30_diag_harness.py gae+rmfix+tiesdrop+norm 30 0

Diagnose why the Day 30 RLHF demo does not learn.

Variants (all on CartPole-v1, same seeds):
  orig      : repo code as-is
  true_nomask: PPO trained on the TRUE reward (+1/step) but with the repo's GAE (no episode-boundary mask)
  true_mask : TRUE reward + done-masked GAE
  rm_fixed  : learned reward, RM fixes (mask padding, both segs padded to common length, ties dropped,
              train even when buffer < batch) + done-masked GAE
Usage: python validation/day30_diag_harness.py <variant> <cycles> [seed]
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, 'papers', '30_RLHF'))
import implementation as I


def ppo_update(tr, rollout_obs, rollout_acts, rollout_logprobs, rollout_vals, rewards, dones, mask_dones):
    steps = len(rewards)
    gamma, lam = 0.99, 0.95
    vals = list(rollout_vals) + [0.0]
    advs = np.zeros(steps)
    gae = 0.0
    for t in reversed(range(steps)):
        nonterm = 1.0 - dones[t] if mask_dones else 1.0
        delta = rewards[t] + gamma * vals[t + 1] * nonterm - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        advs[t] = gae
    returns = torch.tensor(advs + np.array(vals[:-1]), dtype=torch.float32)
    advantages = torch.tensor(advs, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    b_obs = torch.stack(rollout_obs); b_acts = torch.stack(rollout_acts); b_lp = torch.stack(rollout_logprobs)
    for _ in range(4):
        _, new_lp, ent, new_v = tr.policy.get_action_and_value(b_obs, b_acts)
        ratio = torch.exp(new_lp - b_lp)
        pl = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        vl = F.mse_loss(new_v.squeeze(-1), returns)
        loss = pl + 0.5 * vl - 0.01 * ent.mean()
        tr.policy_opt.zero_grad(); loss.backward(); tr.policy_opt.step()


def rollout(tr, steps, reward_fn):
    obs, _ = tr.env.reset()
    O, A, LP, V, R, D = [], [], [], [], [], []
    for _ in range(steps):
        o = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a, lp, _, v = tr.policy.get_action_and_value(o)
        nxt, true_r, term, trunc, _ = tr.env.step(a.item())
        O.append(o); A.append(a); LP.append(lp); V.append(v.item())
        R.append(reward_fn(o, true_r)); D.append(float(term or trunc))
        obs = nxt
        if term or trunc:
            obs, _ = tr.env.reset()
    return O, A, LP, V, R, D


def train_rm_fixed(tr, epochs=5, batch_size=32):
    buf = tr.preference_buffer
    if len(buf) == 0:
        return float('nan'), float('nan')
    bs = min(batch_size, len(buf))
    idx = np.arange(len(buf)); tl = ta = n = 0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx) - bs + 1, bs):
            batch = [buf[j] for j in idx[i:i + bs]]
            L = max(max(len(b[0]), len(b[1])) for b in batch)
            def pad(a):
                out = np.zeros((L, a.shape[1]), np.float32); out[:len(a)] = a
                m = np.zeros(L, np.float32); m[:len(a)] = 1
                return out, m
            p1 = [pad(b[0]) for b in batch]; p2 = [pad(b[1]) for b in batch]
            s1 = torch.tensor(np.stack([p[0] for p in p1])); m1 = torch.tensor(np.stack([p[1] for p in p1]))
            s2 = torch.tensor(np.stack([p[0] for p in p2])); m2 = torch.tensor(np.stack([p[1] for p in p2]))
            y = torch.tensor([float(b[2]) for b in batch], dtype=torch.float32)
            r1 = (tr.reward_model(s1).squeeze(-1) * m1).sum(1)
            r2 = (tr.reward_model(s2).squeeze(-1) * m2).sum(1)
            logits = torch.stack([r1, r2], 1)
            loss = F.cross_entropy(logits, torch.stack([1 - y, y], 1))
            tr.rm_opt.zero_grad(); loss.backward(); tr.rm_opt.step()
            tl += loss.item(); keep = y != 0.5
            ta += ((logits.argmax(1).float() == y)[keep]).float().mean().item() if keep.any() else 1.0; n += 1
    return tl / max(n, 1), ta / max(n, 1)


def evaluate(tr, env, n=10, seed0=10_000):
    out = []
    for k in range(n):
        obs, _ = env.reset(seed=seed0 + k); R = 0
        while True:
            with torch.no_grad():
                a, _, _, _ = tr.policy.get_action_and_value(torch.tensor(obs, dtype=torch.float32))
            obs, r, d, t, _ = env.step(a.item()); R += r
            if d or t: break
        out.append(R)
    return float(np.mean(out))


def main():
    variant = sys.argv[1]; cycles = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    flags = set(variant.split('+'))
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make('CartPole-v1'); env.reset(seed=seed); env.action_space.seed(seed)
    eval_env = gym.make('CartPole-v1')
    tr = I.RLHF_Trainer(env, 4, 2, segment_length=50)
    t0 = time.time(); hist = []
    for c in range(cycles):
        n0 = len(tr.preference_buffer)
        tr.collect_and_label_data(10)
        new = tr.preference_buffer[n0:]
        if 'tiesdrop' in flags:
            new = [p for p in new if len(p[0]) != len(p[1])]
        elif 'tieshalf' in flags:
            new = [(a, b, 0.5) if len(a) == len(b) else (a, b, l) for a, b, l in new]
        tr.preference_buffer = tr.preference_buffer[:n0] + new
        if 'rmfix' in flags:
            rml, rma = train_rm_fixed(tr, epochs=5)
        else:
            rml, rma = tr.train_reward_model(epochs=5)
        fn = lambda o, r: tr.reward_model(o).item()
        with torch.no_grad():
            data = list(rollout(tr, 1000, fn))
        if 'norm' in flags:
            R = np.array(data[4]); data[4] = list((R - R.mean()) / (R.std() + 1e-8))
        ppo_update(tr, *data, mask_dones=('gae' in flags))
        ev = evaluate(tr, eval_env)
        hist.append(ev)
        if (c + 1) % max(1, cycles // 5) == 0 or c == cycles - 1:
            print(f'{variant} cycle {c+1:3d} | RM loss {rml:.3f} acc {rma:.2f} | buffer {len(tr.preference_buffer)} | '
                  f'true reward {ev:6.1f} | {time.time()-t0:5.1f}s', flush=True)
    print(f'RESULT {variant} seed {seed}: first-3 avg {np.mean(hist[:3]):.1f}, last-5 avg {np.mean(hist[-5:]):.1f}, time {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
