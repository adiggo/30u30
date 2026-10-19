"""Day 05 exercises: run each exercise's test_implementation() with the reference solutions
swapped into the exercise module (functions/methods the tests call), and parse PASS/FAIL lines.
Functions that have no reference solution are left as the exercise stubs and reported.
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings
import numpy as np

warnings.simplefilter('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '05_MDL_Principle')
EX = os.path.join(DAY, 'Exercises')
sys.path.insert(0, EX)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EX, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load('solutions')
swaps = {
    'exercise_01_two_part_code': {
        'universal_code_length': S.solution_universal_code_length,
        'model_description_length': S.solution_model_description_length,
        'data_description_length': S.solution_data_description_length,
        'two_part_mdl': S.solution_two_part_mdl},
    'exercise_02_prequential': {
        'predict_next_point': S.solution_predict_next_point,
        'code_length_for_point': S.solution_code_length_for_point,
        'prequential_mdl': S.solution_prequential_mdl,
        'prequential_model_selection': S.solution_prequential_model_selection},
    'exercise_03_model_selection': {
        'compare_with_cross_validation': S.solution_compare_with_cross_validation,
        'universal_code_length': S.solution_universal_code_length,
        'compute_mdl': S.solution_two_part_mdl},
    'exercise_04_nml_complexity': {   # [PATCH 2026-10-18] redesigned exercise 4 (restricted NML + RNML)
        'nml_complexity_polynomial': S.solution_nml_complexity_polynomial,
        'estimate_nml_complexity_polynomial': S.solution_estimate_nml_complexity_polynomial,
        'compute_nml_mdl': S.solution_compute_nml_mdl,
        'compare_nml_vs_two_part': S.solution_compare_nml_vs_two_part,
        'visualize_complexity_vs_degree': S.solution_visualize_complexity_vs_degree,
        'logsumexp': S.solution_logsumexp,
        'gaussian_log_likelihood': S.solution_gaussian_log_likelihood},
    'exercise_05_mdl_vs_aic_bic': {
        'compute_aic': S.solution_compute_aic,
        'compute_bic': S.solution_compute_bic,
        'compute_mdl': lambda x, y, d, precision=32: S.solution_two_part_mdl(x, y, d, precision)[0],
        'select_with_all_criteria': S.solution_select_with_all_criteria,
        'monte_carlo_comparison': S.solution_monte_carlo_comparison,
        # [PATCH 2026-10-18] the three analyses now have reference solutions
        'analyze_sample_size_effect': S.solution_analyze_sample_size_effect,
        'analyze_noise_level_effect': S.solution_analyze_noise_level_effect,
        'when_do_they_disagree': S.solution_when_do_they_disagree},
}
summary = {}
for name, fns in swaps.items():
    mod = load(name)
    for k, v in fns.items():
        setattr(mod, k, v)
    if name == 'exercise_03_model_selection':
        # the tests call methods of the exercise class; take them from the solution class
        for meth in ('fit', 'predict', 'get_compression_ratio', 'get_model_probability'):
            setattr(mod.MDLModelSelector, meth, getattr(S.SolutionMDLModelSelector, meth))
        mod.solution_two_part_mdl = S.solution_two_part_mdl
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.test_implementation()
    out = buf.getvalue()
    lines = [l.strip() for l in out.splitlines() if ('PASSED' in l or 'FAILED' in l or 'NOTE' in l
                                                    or 'Skipped' in l)]
    summary[name] = lines
    print(f"=== {name}")
    for l in lines:
        print("   ", l)
print()
for name, lines in summary.items():
    nfail = sum('FAILED' in l for l in lines)
    print(f"{name}: {sum('PASSED' in l for l in lines)} passed, {nfail} failed, "
          f"{sum(('NOTE' in l) or ('Skipped' in l) for l in lines)} notes/skips")
