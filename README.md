Imperial ML & AI: Black-Box Optimisation Capstone

Capstone project for the Professional Certificate in Machine Learning and Artificial Intelligence, Imperial College Business School.

1. Project overview

Eight unknown functions. Each takes a vector of numbers and returns a single number. No formula, no gradient, no way to inspect what happens inside. The only way to learn anything is to query a point and see what comes back, at one query per function per week across Modules 12 to 24, roughly thirteen in total. The goal is to find the input that maximises each function within that budget.

This matters because the same shape of problem appears whenever evaluation is expensive. Tuning a model's hyperparameters is exactly this: the relationship between settings and performance is unknown, each trial costs time or money, and you cannot try everything. The same holds for clinical trial design and chemical process optimisation.

My interest is machine learning for venture investing, where the equivalent question is which candidate deserves the next expensive piece of attention. The transferable skill is not the Gaussian process itself but the discipline of making every costly evaluation answer a question stated in advance.

2. Inputs and outputs

Every input value lies in [0, 1]. Queries are submitted as a dash-separated string with each value given to exactly six decimal places, and each value must begin with 0, so 1.000000 is invalid and 0.999999 is the effective upper bound.

Example, a 4D query:

0.001000-0.800000-0.999999-0.999999

Each query returns a single scalar. The functions differ substantially in dimension, starting data and output scale:

Fn	Dim	Initial pts	Output character
1	2	10	tiny positives, 200+ orders of magnitude
2	2	10	0 to 0.65, noisy
3	3	15	negative, ceiling 0
4	4	30	negative, ceiling 0
5	4	20	0.1 to 2650, skewed
6	5	20	negative, ceiling 0
7	6	30	0 to 1.8
8	8	40	9 to 10

Results arrive later in the week, so each round is decided without feedback on the current one.

3. Challenge objectives

All eight are maximisation problems. Where the natural objective is a minimisation, such as drug side effects or a recipe penalty, the output is negated so higher is always better.

The binding constraints are the query budget, the delay before results return, and the absence of structural information. Grid search is impossible: ten points per dimension would need a hundred million evaluations in eight dimensions. Every query has to be justified.

4. Technical approach

Rounds 1 to 3. I fit a Gaussian process surrogate to each function and use expected improvement to propose candidates. Two adjustments matter. Where outputs are skewed I fit on log(y), which is monotone so the best input is unchanged but the model is no longer dominated by one extreme reading. Where outputs are noisy I add a noise term, which also estimates how large a difference must be to mean anything.

Alongside the model I compute a second candidate by repeating whatever step last produced a gain, and treat agreement between the two as a confidence score. Where they agree I act confidently; where they diverge the surrogate is usually unreliable and I fall back on structure.

I also use soft-margin SVMs to classify the best third of results against the rest, then restrict the search to the region the classifier rates highly. This works best where the surrogate is weakest: in eight dimensions with 42 points a linear boundary separates good from bad at 95 per cent cross-validated accuracy, so while I cannot predict values I can rule out most of the space.

I also test hypotheses drawn from each function's description. For the drug combination, balance between doses predicts the outcome better than any single compound; for the recipe, the combined sugar and milk load matters more than any single ingredient.

Balance. Roughly a quarter of queries are exploratory, concentrated where the description warns of multiple optima or where a region is unmapped. With ten queries left per function, a wasted round costs a tenth of the remaining budget.

This section is a living record and will be updated as the approach evolves.

Repo layout
stage1/              practice notebooks and reflections
  reflections/       Stage 1 discussion write-ups
stage2/              BBO analysis notebook, query log, results
docs/                datasheet, model card, non-technical write-up
README.md
Stage 1 context (Modules 3 to 11)

Before the BBO challenge, Stage 1 covered skill-building exercises on a practice dataset of my choosing plus the built-in Wine dataset. I used Home Credit Default Risk (Kaggle): predicting whether a thin-file loan applicant will repay, from application and credit-history data. Binary classification, imbalanced (defaults around 8 per cent), scored on ROC-AUC. That work lives in stage1/ and is not part of the graded Stage 2 challenge.

Tools

Python, NumPy, scikit-learn (GaussianProcessRegressor, SVC), SciPy, Matplotlib, Jupyter.
