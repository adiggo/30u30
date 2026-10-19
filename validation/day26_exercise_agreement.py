"""Day 26 exercise <-> solution agreement: day06_exercise_agreement.run_day on papers/26_kolmogorov_complexity.

Run:  python validation/day26_exercise_agreement.py
"""
import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for day06_exercise_agreement
import matplotlib
matplotlib.use("Agg")
from day06_exercise_agreement import run_day
run_day(os.path.join(REPO, "papers", "26_kolmogorov_complexity"))
