# Exercises: MDL Principle

5 exercises implementing concepts from Grünwald (2004). Each exercise maps directly to a section of the paper.

---

## Exercise Overview

| # | Exercise | Difficulty | Paper Section | Source |
|---|----------|------------|--------------|--------|
| 1 | Two-Part Code | Easy | Secs. 1.3 and 2.4 (crude MDL) | Paper concept, our implementation |
| 2 | Prequential MDL | Medium | Sec. 2.6.4 (prequential interpretation) | Paper concept, our implementation |
| 3 | Model Selection | Medium | Example 1.2 (polynomial) | Paper's own example |
| 4 | NML Complexity | Hard | Secs. 2.5.3, 2.6, 2.7.2, 2.8 (NML, COMP, infinite COMP) | Paper concept; closed forms derived/checked by us and Rissanen (2000) |
| 5 | MDL vs AIC vs BIC | Medium | Sec. 2.9.2 (MDL vs BIC; AIC is not treated) | Our comparison and Monte Carlo extension |

---

## How to Work Through These

1. Read `paper_notes.md` for the theory
2. Check `CHEATSHEET.md` for formulas
3. Each exercise file has TODO markers — fill them in
4. Run the file to check against built-in tests

```bash
python exercise_01_two_part_code.py
```

Passing tests show `[ok] PASSED`. Look at `solutions.py` only after attempting each exercise.

---

## Exercise 1: Two-Part Code [Easy]

**Paper Secs. 1.3 and 2.4.** Implement the basic MDL score: L(H) + L(D|H).

**Tasks:**
- Compute model description length (parameters x bits)
- Compute data description length (Gaussian residual code)
- Select the polynomial degree with minimum total score

---

## Exercise 2: Prequential MDL [Medium]

**Paper Sec. 2.6.4.** Implement sequential prediction coding.

**Tasks:**
- Predict each data point using only previous points
- Accumulate log-loss (cumulative surprise)
- Compare to two-part MDL scores

---

## Exercise 3: Model Selection [Medium]

**Paper's Example 1.2.** Full polynomial degree selection with MDL.

**Tasks:**
- Generate noisy polynomial data (known true degree)
- Compute MDL scores for degrees 1 through 10
- Verify MDL selects the true degree
- Visualize the model-vs-data cost tradeoff

---

## Exercise 4: NML Complexity [Hard]

**Paper Secs. 2.5.3, 2.6, 2.7.2 and 2.8.** Plain NML is undefined for Gaussian regression (its complexity is infinite, Example 2.20), so you work with a restricted model.

**Tasks:**
- Implement the closed-form complexity COMP of the polynomial model with known sigma and the fitted curve's mean square bounded by R; check that it is >= 0, grows with the degree, and grows by (k/2) log2 of the ratio when R grows
- Check it with a Monte Carlo estimate on a tiny design, including the sampling box's volume
- Select the degree with Rissanen's renormalised NML (unknown sigma) and compare with the two-part code

---

## Exercise 5: MDL vs AIC vs BIC [Medium]

**Paper Sec. 2.9.2 (BIC); AIC is background.** Head-to-head comparison.

**Tasks:**
- Implement AIC, BIC, and MDL scoring
- Run polynomial selection with all three criteria
- Monte Carlo: compare accuracy across many random datasets (our extension)
- Analyze when the criteria agree and disagree

---

Solutions in `solutions.py`.
