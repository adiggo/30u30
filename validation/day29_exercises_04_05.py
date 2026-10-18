"""Day 29: run exercise_04 (epsilon ablation) and exercise_05 (PPO vs vanilla PG, solution swapped in)
to check the solutions' 'expected findings' (e.g. 'epsilon=0.2 should reach CartPole reward > 400 within
50 iterations').
Run:  OMP_NUM_THREADS=1 python validation/day29_exercises_04_05.py 04|05   (slow: trains CartPole agents)"""
import os, sys, importlib
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '29_PPO')
sys.path.insert(0, DAY); sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
part = sys.argv[1]
if part == '05':
    ex = importlib.import_module('exercises.exercise_05_ppo_vs_pg')
    ex.vanilla_pg_loss = importlib.import_module('solution_05_ppo_vs_pg').vanilla_pg_loss
    res = ex.run_comparison(env_name='CartPole-v1', n_iterations=80)
    for k, v in res.items():
        print(f'{k}: final-10 avg {np.mean(v[-10:]):.1f}, max {max(v):.1f}, iterations >= 400: {sum(r >= 400 for r in v)}')
else:
    ex = importlib.import_module('exercises.exercise_04_epsilon_ablation')
    res = ex.run_epsilon_ablation(env_name='CartPole-v1', n_iterations=50, epsilon_values=[0.05, 0.1, 0.2, 0.3, 0.5])
    ex.analyze_results(res)
