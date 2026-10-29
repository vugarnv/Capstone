# Datasheet: BBO Capstone Query Data Set

Following the datasheet framework of Gebru et al. (Datasheets for Datasets), as taught in Mini-lesson 21.1. Final version, after round 13 (Module 25, October 2026).

## 1. Motivation

**Why was this data set created?** To support the Stage 2 black-box optimisation (BBO) challenge of the Imperial College Business School Professional Certificate in ML and AI. Eight unknown functions had to be maximised under a hard budget of one query per function per week over thirteen weeks (Modules 12 to 24). The data set is the complete record of every point I queried, every value the portal returned, and the reason each point was chosen.

**What task does it support?** Sequential model-based optimisation: fitting a surrogate (a Gaussian process, with a support-vector classifier as a feasibility filter on the higher-dimensional functions) to past evaluations and choosing the next query by expected improvement. It also supports a secondary task: auditing my own decisions, which queries the model chose, which I overrode, and what each cost or gained.

**Who created it, and was it funded?** Created by me (Vugar Nasibov) as a course participant. The initial points for each function were supplied by the course team; every subsequent point is mine. No external funding.

## 2. Composition

**What do the instances represent?** One instance is a single evaluation of one function: a function id (1 to 8), a round number (0 for the course-supplied starting data, 1 to 13 for my weekly submissions), an input vector in [0, 1]^d with d between 2 and 8, the scalar output the portal returned, and a decision tag recording what chose the point.

**How many instances?** The starting data gave 175 points across the eight functions (10, 10, 15, 30, 20, 20, 30 and 40). Thirteen rounds added 104 more, so the data set holds 279 evaluations. Per function that is between 23 (functions 1 and 2) and 53 (function 8).

**Is it a sample or complete?** Complete with respect to my own queries. It is a very sparse sample of each function's domain: 23 points in a two-dimensional unit square is thin, and 53 points in an eight-dimensional unit cube is thinner than it sounds, since the volume grows as the eighth power of the side.

**Format.** `stage2/query_log.csv`, one row per evaluation with columns `function, round, x1 ... x8 (unused dimensions blank), y, decided_by`. Inputs are stored exactly as submitted, six decimal places, so the file reproduces the portal strings; outputs exactly as returned. The course-supplied starting points are in `stage2/initial_data/function_k/initial_inputs.npy` and `initial_outputs.npy`. The analysis notebook `stage2/bbo_weekly_analysis.ipynb` holds the same history in its `ROUNDS` list and re-derives every model from it.

**The functions, and why their outputs must never be pooled:**

| Fn | Dim | Points after round 13 | What the course says it represents | Output character | Starting best | Final best |
|---|---|---|---|---|---|---|
| 1 | 2 | 23 | locating a radiation source | tiny positives spanning 200+ orders of magnitude; analysed on log scale; deterministic | 7.7 × 10⁻¹⁶ | 1.998 |
| 2 | 2 | 23 | a noisy log-likelihood with local optima | 0 to 0.65, noisy; repeated readings differ | 0.611 | 0.643 |
| 3 | 3 | 28 | adverse reactions to a three-compound drug (negated) | negative, ceiling 0; shown noisy in round 13 (same input, −0.009 then −0.024) | −0.035 | −0.0089 |
| 4 | 4 | 43 | warehouse-placement model, four hyperparameters | negative at start, positive from round 3; deterministic | −4.03 | 0.682 |
| 5 | 4 | 33 | yield of a chemical process | 0.1 to 8,700, heavily skewed; analysed on log scale; deterministic | 1,089 | 8,662 |
| 6 | 5 | 33 | cake recipe penalty, five ingredients (negated) | negative, ceiling 0 | −0.714 | −0.203 |
| 7 | 6 | 43 | ML model with six hyperparameters | 0 to 3.2 | 1.365 | 3.162 |
| 8 | 8 | 53 | ML model with eight hyperparameters | 9 to 10, narrow band; deterministic | 9.598 | 9.983 |

**Gaps and errors.** Functions 2 and 3 are noisy, so some of their variation is measurement, not signal; the round 13 repeat on function 3 is the direct evidence. Several queries sit on the domain edge (0.000000 or 0.999999), either because the acquisition function found nothing better inside or, on function 5, because the measured trend led to the corner; they are kept and tagged. The sampling is not uniform: after round 3, queries concentrate in the regions the surrogate or the measured trend rated highly, so the data set over-represents good regions and under-represents the rest. There are no missing outputs; every submitted query was answered. Five rows are exact repeats of an earlier query (round 13 on functions 1, 4, 5 and 8, and round 12/13 on function 8), kept because the repeated value is itself evidence about noise.

**Decision tags.** `surrogate` (GP with expected improvement), `surrogate-log` (same on log y), `svm-filter` (GP restricted to the SVM fence), `structural` (hypothesis from the function description), `trend` (extrapolating a measured trend after the model lost its vote), `exploration` (deliberate probe of an unsampled region), `repeat` (incumbent re-queried to certify it). The tags were reconstructed in Module 25 from the weekly reflections rather than logged at the time, so they are a faithful summary of the decision process but not a contemporaneous record.

**Relationships between instances.** Rows for the same function are a time series of one optimisation run; row order is decision order. There are no relationships across functions.

**Recommended splits.** None for the optimisation task, since every point informs the next query. For validating a surrogate I used leave-one-out cross-validation within a function, never a random split across rounds, because later points depend on earlier ones.

**Privacy and people.** The data set contains no people, no personal data and no sub-populations.

## 3. Collection process

**How was the data acquired?** By submitting one query string per function per week to the course's capstone portal and recording the scalar it returned later that week, from the portal's confirmation email. The functions are synthetic and their form is hidden; the only instrument is the portal.

**Sampling strategy.** Deterministic and model-based, not random. Rounds 1 to 3: Gaussian process with expected improvement, on log-transformed outputs where skewed and with a fitted noise term where noisy. Rounds 4 to 7: the same surrogate plus an SVM filter that fences off the region rated in the top third of results, a second candidate produced by repeating the last successful step, and structural hypotheses drawn from each function's description (dose balance for the drug function, combined sugar and milk load for the recipe). Rounds 8 to 13: a multi-seed stability check and a margin test on every proposal, and the rule that a function which beat its forecast twice in the same direction lost its model vote, after which the measured trend was extrapolated (functions 1, 5 and 7). Roughly a fifth of queries were deliberately exploratory. Round 13 re-queried the incumbent on four functions to certify it.

**Time frame.** Starting data supplied in Module 12 (July 2026). Rounds 1 to 13 submitted weekly from 24 July to 22 October 2026, with the programme's two break weeks.

**Ethical review, consent.** Not applicable; no human subjects. The functions are described by the course as analogues of real tasks, but the numbers are synthetic.

## 4. Preprocessing and uses

**Transformations applied.** Raw outputs are stored unchanged. For modelling, functions 1, 5 and 7 are fitted on log(y), which is monotone and so leaves the location of the maximum unchanged; the log values are computed on the fly and not stored. Function 1's one negative starting value (−0.0036) is floored before the log, since it is effectively zero on that scale. Inputs are used as submitted; no scaling is needed since all lie in [0, 1]. The raw data is therefore always preserved.

**Intended uses.** Fitting surrogates to choose the next query; auditing the decision history; comparing acquisition strategies on the same record; the final capstone report.

**Inappropriate uses.** Estimating the global maximum of any function from these points alone, since the sampling is biased toward regions already thought good. Pooling outputs across functions, since scales differ by hundreds of orders of magnitude. Treating a single reading of function 2 or 3 as exact. Generalising anything about the real-world processes the functions are named after.

## 5. Distribution and maintenance

**Where is it available?** In the public GitHub repository `github.com/vugarnv/Capstone`, under `stage2/`, with the analysis notebook beside it.

**Terms of use.** MIT licence for my own rows and code (see `LICENSE`). The course-supplied starting points remain the property of the programme and are included for reproducibility only.

**Who maintains it?** I do. The challenge is complete and the data set is final; no further rows will be added. Errors found later will be corrected in place with a note in the commit message.
