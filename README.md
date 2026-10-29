# Imperial ML and AI: Black-Box Optimisation Capstone

Capstone project for the Professional Certificate in Machine Learning and Artificial Intelligence, Imperial College Business School. Thirteen query rounds, Modules 12 to 24 (July to October 2026).

## What this project did, in plain terms

I was given eight hidden formulas. Each takes a few numbers in and gives one number back, and the only way to learn about a formula was to try an input and see the result, once a week for thirteen weeks. The aim was to find the input that gives the highest result for each one. My approach was to build a statistical guess of each formula from the results so far, use that guess to choose the next try, and check every week whether the guess was earning its keep. Where it was not, I followed the measured trend instead. All eight functions ended above their starting best; the same method applies wherever each experiment is expensive.

## 1. Project overview

Eight unknown functions. Each takes a vector of numbers in [0, 1] and returns a single number. No formula, no gradient, no way to inspect what happens inside. The only way to learn anything is to query a point and see what comes back, at one query per function per week, thirteen in total. The goal is the input that maximises each function within that budget.

This matters because the same shape of problem appears whenever evaluation is expensive. Tuning a model's hyperparameters is exactly this: the relationship between settings and performance is unknown, each trial costs time or money, and you cannot try everything. The same holds for clinical trial design and chemical process optimisation.

My interest is machine learning for venture investing, where the equivalent question is which candidate deserves the next expensive piece of attention. The transferable skill is not the Gaussian process itself but the discipline of making every costly evaluation answer a question stated in advance, and of knowing when a model has stopped earning its vote.

## 2. Inputs and outputs

Every input value lies in [0, 1]. Queries are submitted as a dash-separated string with each value given to exactly six decimal places, and each value must begin with 0, so 1.000000 is invalid and 0.999999 is the effective upper bound.

Example, a 4D query: `0.001000-0.800000-0.999999-0.999999`

Each query returns a single scalar. The functions differ substantially in dimension, starting data and output scale:

| Fn | Dim | Initial pts | What the course says it represents | Output character |
|---|---|---|---|---|
| 1 | 2 | 10 | locating a radiation source | tiny positives, 200+ orders of magnitude |
| 2 | 2 | 10 | a noisy log-likelihood with local optima | 0 to 0.65, noisy |
| 3 | 3 | 15 | adverse reactions to a three-compound drug (negated) | negative, ceiling 0, noisy |
| 4 | 4 | 30 | warehouse-placement model, four hyperparameters | negative at start, ceiling near 0.7 |
| 5 | 4 | 20 | yield of a chemical process | 0.1 to 8,700, heavily skewed |
| 6 | 5 | 20 | cake recipe penalty, five ingredients (negated) | negative, ceiling 0 |
| 7 | 6 | 30 | ML model with six hyperparameters | 0 to 3.2 |
| 8 | 8 | 40 | ML model with eight hyperparameters | 9 to 10, narrow band |

Results arrive later in the week, so each round is decided without feedback on the current one.

## 3. Challenge objectives

All eight are maximisation problems. Where the natural objective is a minimisation, such as drug side effects or a recipe penalty, the output is negated so higher is always better.

The binding constraints are the query budget, the delay before results return, and the absence of structural information. Grid search is impossible: ten points per dimension would need a hundred million evaluations in eight dimensions. Every query has to be justified.

## 4. Technical approach

The pipeline went through three versions. Everything is re-derived from the query history in the notebook, so any round can be replayed with the data available at the time.

**Version 1, rounds 1 to 3.** A Gaussian process surrogate (Matern 2.5 kernel) with expected improvement, one function at a time. Two fixes came out of round 1. Where outputs are skewed (functions 1 and 5) the GP is fitted on log(y), which is monotone, so the best input is unchanged but the model is no longer dominated by one extreme reading. Where outputs are noisy (function 2) a noise term is fitted, which also estimates how large a difference must be to mean anything. Round 1 beat the starting best on six of the eight functions; function 1 did not until the log fix, and function 3 came back below it.

**Version 2, rounds 4 to 7.** Three additions. A soft-margin SVM trained to separate the top third of results from the rest, used as a fence: in eight dimensions with forty points no surrogate could predict function 8, but a linear boundary separated good from bad at high cross-validated accuracy, so the search could rule out most of the cube and spend every query inside the fence. A second candidate made by repeating whatever step last produced a gain, with agreement between it and the GP treated as a confidence score. And structural hypotheses drawn from each function's description: for the drug function the balance between the three doses predicted the outcome better than any single compound; for the recipe the combined sugar and milk load mattered more than any single ingredient.

**Version 3, rounds 8 to 13.** The control layer. Each GP is refitted across several random seeds and a proposal that moves with the seed is discarded. A margin test drops any proposal whose predicted gain over the incumbent is thin or negative. And a rule that a function which beats its forecast twice in the same direction loses its model vote, after which the measured trend is extrapolated instead. Function 5 triggered that rule in rounds 7 and 8 and the corner of the box found in round 9 (8,662) was the result. Round 13 was spent certifying: the incumbent was re-queried on functions 1, 4, 5 and 8 and returned the identical value each time, which established those surfaces as deterministic; the same test on function 3 returned a different value and established it as noisy.

**Balance.** Roughly a fifth of queries were exploratory, concentrated where the description warned of multiple optima or where a region was unmapped. From round 11 the exploration query on the ceiling-at-zero functions was placed in the least-sampled quadrant rather than at the GP's highest-variance point, because the two differ when the model is wrong.

## 5. Results

Best observed value per function, from the course-supplied starting data through the thirteen rounds. Every query and result is in `stage2/query_log.csv`; the notebook reproduces this table in its final section.

| Fn | Dim | Starting best | Best after round 10 | Final best | Round | What chose the winning query |
|---|---|---|---|---|---|---|
| 1 | 2 | 7.7e-16 | 1.980 | 1.998 | 12 | long strides along a ridge measured on the log scale |
| 2 | 2 | 0.611 | 0.643 | 0.643 | 1 | GP with expected improvement; never beaten, noise exceeded any predicted gain |
| 3 | 3 | −0.035 | −0.0135 | −0.0089 | 11 | structural hypothesis (dose balance); function later shown to be noisy |
| 4 | 4 | −4.03 | 0.682 | 0.682 | 4 | GP with expected improvement; incumbent re-queried in round 13, identical value |
| 5 | 4 | 1,089 | 8,662 | 8,662 | 9 | measured trend to the corner of the box after the model lost its vote |
| 6 | 5 | −0.714 | −0.238 | −0.203 | 12 | GP with expected improvement once conditioned on the full history |
| 7 | 6 | 1.365 | 2.571 | 3.162 | 12 | moving the first input off the edge the surrogate had held it at since round 1 |
| 8 | 8 | 9.598 | 9.972 | 9.983 | 12 | GP with expected improvement, the model-led plan working as intended |

Five of the eight functions produced a new best in the final three rounds; round 11 was the strongest single round, with new bests on functions 1, 3, 6, 7 and 8 at once. Of the 104 queries, 47 were chosen by the surrogate, 20 by extrapolating a measured trend, 19 were deliberate exploration, 8 came from structural hypotheses, 5 from the SVM fence and 5 were repeats to certify an incumbent.

What I would do differently: look at every function on a log scale before fitting anything; explore the ceiling-at-zero functions harder in the early rounds; run the SVM fence from round 1; and test the simplest hypothesis first, since a peer fitted function 8 exactly with a 17-parameter quadratic from the starting data alone.

## 6. Documentation

- Datasheet for the query data set (motivation, composition, collection, uses, maintenance): [docs/datasheet.md](docs/datasheet.md)
- Model card for the optimisation approach (intended use, evaluation, assumptions, failure modes): [docs/model_card.md](docs/model_card.md)

Both follow the frameworks from Module 21 (Gebru et al., Datasheets for Datasets; Mitchell et al., Model Cards for Model Reporting) and are finalised at round 13.

## Repository layout

```
README.md
LICENSE                          MIT
docs/
  datasheet.md                   datasheet for the query data set
  model_card.md                  model card for the optimiser
stage2/
  bbo_weekly_analysis.ipynb      the analysis notebook, full history in its top cell, executed
  query_log.csv                  every query and result, 104 rows, one per function per round
  initial_data/function_1..8/    the course-supplied starting points (.npy)
stage1/
  reflections/                   Stage 1 discussion write-ups (Modules 6 to 8)
```

## How to run

```
pip install numpy scipy scikit-learn matplotlib jupyter
jupyter notebook stage2/bbo_weekly_analysis.ipynb
```

Run all cells. The notebook reads `stage2/initial_data/` and the `ROUNDS` list in section 1; no other data is needed. To replay an earlier round, shorten `ROUNDS` to the first *k* entries and re-run.

## Stage 1 context (Modules 3 to 11)

Before the BBO challenge, Stage 1 covered skill-building exercises on a practice dataset of my choosing plus the built-in Wine dataset. I used Home Credit Default Risk (Kaggle): predicting whether a thin-file loan applicant will repay, from application and credit-history data. Binary classification, imbalanced (defaults around 8 per cent), scored on ROC-AUC. That work lives in `stage1/` and is not part of the graded Stage 2 challenge.

## Tools

Python 3, NumPy, scikit-learn (GaussianProcessRegressor, SVC), SciPy, Matplotlib, Jupyter.

## Contact

Vugar Nasibov, via this repository.
