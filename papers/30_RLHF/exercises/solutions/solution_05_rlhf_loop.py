"""
solution_05_rlhf_loop.py - Day 30: RLHF
Goal: Orchestrate the full RLHF loop (Collect -> Label -> Train RM -> Train Policy).

Christiano et al. (2017, Sec. 2.2) run three processes asynchronously: the policy acts in the
environment and is optimised against the current reward estimate r_hat (Sec. 2.2.1), pairs of
trajectory segments go to a human for comparison (Sec. 2.2.2), and r_hat is fitted to all
comparisons so far with the Bradley-Terry loss (Sec. 2.2.3). This solution runs them one after
another in cycles, using the repo's RLHF_Trainer (implementation.py) for each phase:

    1-2. collect_and_label_data()  roll out the policy, cut 50-step segments, the SyntheticOracle
                                   prefers the segment with more TRUE reward (ties = 0.5)
    3.   train_reward_model()      fit r_hat(s) to every label in the buffer
    4.   train_policy_ppo()        1,000 environment steps of PPO on r_hat only (the paper used
                                   A2C/TRPO); the true reward is never shown to the policy
    5.   evaluate                  mean TRUE return over a few episodes: did aligning to the
                                   labels improve what the labeller cares about?

Before training on the new pairs, it also measures the reward model's accuracy on them: a
held-out check of whether r_hat generalises to the policy's new behaviour.

Caveat, measured by the control run at the end: in CartPole the true reward is +1 per step, so a
label only says which segment survived longer, and any r_hat that is positive on average already
agrees with every decided label (even the untrained one may score 100%). The policy improves
because each extra step earns positive reward until the episode terminates. The control trains
the same PPO on a constant reward of +1 with no labels at all. Christiano et al. remove such
termination signals from their tasks (Appendix A: they "encode information about the task even
when the reward function is not observable"); here they are the whole signal.

Run (about 1 minute on one CPU core; --no-control halves it):
    python exercises/solutions/solution_05_rlhf_loop.py [--cycles 30] [--seed 0] [--no-control]
A uniformly random CartPole policy scores about 22.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# [PATCH 2026-10-18] runnable loop built from the repo's components (this file used to be
# pseudocode that crashed on env.reset() of None)
DAY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, DAY_DIR)

from implementation import RLHF_Trainer, compute_preference_loss  # noqa: E402

try:
    import gymnasium as gym
except ImportError:  # older installs
    import gym


def pairs_to_batch(pairs):
    """
    Stack (obs1, obs2, label) pairs of possibly different lengths into padded tensors.

    Returns s1, s2: (batch, L, obs_dim); m1, m2: (batch, L) with 1 for real steps, 0 for padding;
    labels: (batch,) floats, P(segment 2 preferred) (0, 1 or 0.5 for a tie).
    """
    L = max(max(len(a), len(b)) for a, b, _ in pairs)
    obs_dim = pairs[0][0].shape[1]

    def pad(seg):
        out = np.zeros((L, obs_dim), dtype=np.float32)
        out[:len(seg)] = seg                                  # copy the real steps, rest stays 0
        mask = np.zeros(L, dtype=np.float32)
        mask[:len(seg)] = 1.0
        return out, mask

    p1 = [pad(a) for a, _, _ in pairs]
    p2 = [pad(b) for _, b, _ in pairs]
    s1 = torch.tensor(np.stack([p[0] for p in p1]))           # np.stack: list of (L, d) -> (B, L, d)
    s2 = torch.tensor(np.stack([p[0] for p in p2]))
    m1 = torch.tensor(np.stack([p[1] for p in p1]))           # (B, L)
    m2 = torch.tensor(np.stack([p[1] for p in p2]))
    labels = torch.tensor([float(lab) for _, _, lab in pairs], dtype=torch.float32)
    return s1, s2, m1, m2, labels


def preference_accuracy(reward_model, pairs):
    """Fraction of decided (non-tie) pairs where r_hat's segment sums agree with the oracle; nan if none."""
    s1, s2, m1, m2, labels = pairs_to_batch(pairs)
    with torch.no_grad():
        _, acc = compute_preference_loss(reward_model, s1, s2, labels, mask1=m1, mask2=m2)
    return acc


def evaluate_true_reward(policy, env, episodes=5, seed=10_000):
    """Mean TRUE return of the current (sampling) policy over a few seeded episodes."""
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(torch.tensor(obs, dtype=torch.float32))
            obs, r, terminated, truncated, _ = env.step(action.item())
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def run_rlhf_step(trainer, eval_env, num_pairs=10, rm_epochs=5, ppo_steps=1000):
    """
    One RLHF cycle. Returns a dict with the reward model's accuracy on the new pairs before training
    on them ('heldout_acc'), its training loss and accuracy ('rm_loss', 'rm_acc'), the mean learned
    reward during the PPO rollout ('learned_reward') and the mean true return ('true_reward').
    """
    # 1-2. Collect segments with the current policy; the oracle labels pairs of them (Sec. 2.2.2)
    trainer.collect_and_label_data(num_new_pairs=num_pairs)
    new_pairs = trainer.preference_buffer[-num_pairs:]

    # How well does the CURRENT r_hat predict labels it has not been trained on?
    heldout_acc = preference_accuracy(trainer.reward_model, new_pairs)

    # 3. Fit r_hat to all comparisons so far (Bradley-Terry cross-entropy, Sec. 2.2.3)
    rm_loss, rm_acc = trainer.train_reward_model(epochs=rm_epochs)

    # 4. Optimise the policy against r_hat only (Sec. 2.2.1)
    learned_reward = trainer.train_policy_ppo(steps=ppo_steps)

    # 5. Measure what we actually care about: the environment's true reward
    true_reward = evaluate_true_reward(trainer.policy, eval_env)
    return {'heldout_acc': heldout_acc, 'rm_loss': rm_loss, 'rm_acc': rm_acc,
            'learned_reward': float(learned_reward), 'true_reward': true_reward}


def make_trainer(seed, env_name='CartPole-v1', segment_length=50):
    """Seeded environment, evaluation environment and RLHF_Trainer (policy, r_hat, oracle, optimizers)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env = gym.make(env_name)
    trainer = RLHF_Trainer(env, env.observation_space.shape[0], env.action_space.n,
                           segment_length=segment_length)
    return trainer, env, eval_env


def run_rlhf(cycles=30, seed=0, env_name='CartPole-v1', segment_length=50, num_pairs=10,
             rm_epochs=5, ppo_steps=1000, verbose=True):
    """Run `cycles` RLHF cycles from scratch; returns the list of per-cycle dicts."""
    trainer, env, eval_env = make_trainer(seed, env_name, segment_length)

    def fmt(x):
        return ' n/a ' if np.isnan(x) else f'{100 * x:4.0f}%'

    history = []
    if verbose:
        print(f"{'cycle':>5s} {'pairs':>6s} {'RM acc (new pairs)':>19s} {'RM acc (train)':>15s} "
              f"{'learned r':>10s} {'true reward':>12s}")
    for cycle in range(1, cycles + 1):
        h = run_rlhf_step(trainer, eval_env, num_pairs, rm_epochs, ppo_steps)
        history.append(h)
        if verbose:
            print(f"{cycle:5d} {len(trainer.preference_buffer):6d} {fmt(h['heldout_acc']):>19s} "
                  f"{fmt(h['rm_acc']):>15s} {h['learned_reward']:10.3f} {h['true_reward']:12.1f}",
                  flush=True)
    env.close()
    eval_env.close()
    return history


class ConstantReward(torch.nn.Module):
    """r_hat(s) = +1 for every state: CartPole's own reward signal, with no labels at all."""

    def forward(self, obs):
        return torch.ones(obs.shape[:-1] + (1,))          # one reward per observation


def run_constant_reward_control(cycles=30, seed=0, env_name='CartPole-v1', ppo_steps=1000):
    """Same seed, policy and PPO phase as run_rlhf, but the reward model is replaced by +1."""
    trainer, env, eval_env = make_trainer(seed, env_name)
    trainer.reward_model = ConstantReward()
    true_rewards = []
    for _ in range(cycles):
        trainer.train_policy_ppo(steps=ppo_steps)
        true_rewards.append(evaluate_true_reward(trainer.policy, eval_env))
    env.close()
    eval_env.close()
    return true_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exercise 5 solution: a small RLHF loop on CartPole')
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-control', action='store_true', help='skip the constant-reward control')
    args = parser.parse_args()

    t0 = time.time()
    hist = run_rlhf(cycles=args.cycles, seed=args.seed)
    true_r = [h['true_reward'] for h in hist]
    held = [h['heldout_acc'] for h in hist if not np.isnan(h['heldout_acc'])]
    print(f"\nRLHF (policy trained on r_hat only): true reward, first 3 cycles {np.mean(true_r[:3]):.1f} "
          f"-> last 5 cycles {np.mean(true_r[-5:]):.1f} (random policy ~22, maximum 500)")
    if held:
        print(f"Reward-model accuracy on new pairs before training on them (decided pairs, all cycles): "
              f"{100 * np.mean(held):.0f}%; cycle 1 used the untrained model: "
              f"{100 * hist[0]['heldout_acc']:.0f}%")
    if not args.no_control:
        control = run_constant_reward_control(cycles=args.cycles, seed=args.seed)
        print(f"Control (same PPO on a constant +1 reward, no labels): first 3 cycles "
              f"{np.mean(control[:3]):.1f} -> last 5 cycles {np.mean(control[-5:]):.1f}")
        print("In CartPole a label only says which segment lasted longer, so the labels carry no more")
        print("than the +1-per-step reward and the episode end; the paper removes that termination")
        print("signal (Appendix A) so that the human labels are the only guidance.")
    print(f"Total time {time.time() - t0:.0f} s.")
