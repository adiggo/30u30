"""
exercise_05_rlhf_loop.py - Day 30: RLHF
Goal: Orchestrate the full RLHF loop (Collect -> Label -> Train RM -> Train Policy).
This is the high-level workflow that aligns the agent to preferences.

[PATCH 2026-10-18] The pieces exist in implementation.py: RLHF_Trainer has one method per phase
(collect_and_label_data, train_reward_model, train_policy_ppo). Your job is the loop: the order of
the phases, what each consumes and produces, and measuring the result with the TRUE reward, which
the policy never sees. (This file used to ask for pseudocode on mock objects; it now runs for real,
like solutions/solution_05_rlhf_loop.py, which also adds a control experiment.)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the day folder
from implementation import RLHF_Trainer  # noqa: E402

try:
    import gymnasium as gym
except ImportError:
    import gym


def evaluate_true_reward(policy, env, episodes=5, seed=10_000):
    """Given: mean TRUE return of the current policy over a few seeded episodes."""
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
    Runs one step (cycle) of the RLHF loop:
    1. Collect trajectory segments with the current policy
    2. Get feedback from the oracle on pairs of segments
    3. Update the reward model on all labels collected so far
    4. Update the policy with PPO against the LEARNED reward
    5. Evaluate the policy with the TRUE reward

    Returns a dict with 'rm_loss', 'rm_acc', 'learned_reward' and 'true_reward'.
    """
    # 1-2. Collect data and get labels
    # TODO: trainer.collect_and_label_data(num_new_pairs=...)

    # 3. Train Reward Model
    # TODO: rm_loss, rm_acc = trainer.train_reward_model(epochs=...)

    # 4. Train Policy
    # TODO: learned_reward = trainer.train_policy_ppo(steps=...)

    # 5. Evaluate
    # TODO: true_reward = evaluate_true_reward(trainer.policy, eval_env)

    raise NotImplementedError("Implement run_rlhf_step!")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make('CartPole-v1')
    env.reset(seed=0)
    eval_env = gym.make('CartPole-v1')
    trainer = RLHF_Trainer(env, env.observation_space.shape[0], env.action_space.n, segment_length=50)

    try:
        for cycle in range(1, 4):
            h = run_rlhf_step(trainer, eval_env)
            print(f"cycle {cycle}: RM accuracy {h['rm_acc']:.0%}, true reward {h['true_reward']:.1f}")
        print("The loop runs. With 30 cycles the solution's true reward climbs from ~26 to ~110.")
    except NotImplementedError as e:
        print(f"Not implemented yet: {e}")
