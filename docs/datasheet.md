# Datasheet: BBO Capstone Query Data Set

Following the datasheet framework of Gebru et al. (Datasheets for Datasets), as taught in Mini-lesson 21.1. Last updated after round 10 (Module 21, October 2026).

## 1. Motivation

**Why was this data set created?** To support the Stage 2 black-box optimisation (BBO) challenge of the Imperial College Business School Professional Certificate in ML and AI. Eight unknown functions must be maximised under a hard budget of one query per function per week over roughly thirteen weeks (Modules 12 to 24). The data set is the complete record of every point I have queried, every value the portal returned, and the reason each point was chosen.

**What task does it support?** Sequential model-based optimisation: fitting a surrogate (a Gaussian process, with a support-vector classifier as a feasibility filter on the higher-dimensional functions) to past evaluations and choosing the next query by expected improvement. It also supports a secondary task: auditing my own decisions, which queries the model chose, which I overrode, and what each cost or gained.

**Who created it, and was it funded?** Created by me (Vugar Nasibov) as a course participant. The initial points for each function were supplied by the course team; every subsequent point is mine. No external funding.

## 2. Composition

**What do the instances represent?** One instance is a single evaluation of one function: a function id (1 to 8), a round number (0 for the course-supplied starting data, 1 to 10 for my weekly submissions), an input vector in [0, 1]^d with d between 2 and 8, the scalar output the portal returned, and a decision tag recording what chose the point (surrogate, structural hypothesis, exploration rule, or repeat of the last successful step).

**How many instances?** The starting data gave 175 points across the eight functions (10, 10, 15, 30, 20, 20, 30 and 40). Ten rounds add 80 more, so the data set holds 255 evaluations after round 10. Per function that is between 20 (functions 1 and 2) and 50 (function 8).

**Is it a sample or complete?** Complete with respect to my own queries. It is a very sparse sample of each function's domain: 20 points in a two-dimensional unit square is thin, and 50 points in an eight-dimensional unit cube is thinner than it sounds, since the volume grows as the eighth power of the side.

**Format.** One row per evaluation with columns `function, round, x1 ... x8 (unused dimensions blank), y, decided_by, note`. Inputs are stored exactly as submitted, six decimal places, so the file reproduces the portal strings. The record of source is the analysis notebook `stage2/bbo_weekly_analysis.ipynb`, whose top cell holds the full history; it is being exported to `stage2/query_log.csv` so the data can be read without running the notebook.

**The functions, and why their outputs must never be pooled:**

| Fn | Dim | Points after round 10 | What the course says it represents | Output character | Starting best |
|---|---|---|---|---|---|
| 1 | 2 | 20 | locating a radiation source | tiny positives spanning 200+ orders of magnitude; analysed on log scale | 7.7 × 10⁻¹⁶ |
| 2 | 2 | 20 | a noisy log-likelihood with local optima | 0 to 0.65, noisy; repeated readings differ | 0.611 |
| 3 | 3 | 25 | adverse reactions to a three-compound drug (negated) | negative, ceiling 0 | −0.035 |
| 4 | 4 | 40 | warehouse-placement model, four hyperparameters | negative, ceiling 0 | −4.03 |
| 5 | 4 | 30 | yield of a chemical process | 0.1 to 2,650, heavily skewed; analysed on log scale | 1,089 |
| 6 | 5 | 30 | cake recipe penalty, five ingredients (negated) | negative, ceiling 0 | −0.714 |
| 7 | 6 | 40 | ML model with six hyperparameters | 0 to 1.8 | 1.365 |
| 8 | 8 | 50 | ML model with eight hyperparameters | 9 to 10, narrow band | 9.598 |

**Gaps and errors.** Function 2 is noisy, so some of its variation is measurement, not signal. Two early queries sat on the domain edge (0.000001 or 0.999999) because the acquisition function found nothing better inside; they are kept but tagged. The sampling is not uniform: after round 3, queries concentrate in the regions the surrogate rated highly, so the data set over-represents good regions and under-represents the rest. There are no missing outputs; every submitted query was answered.

**Relationships between instances.** Rows for the same function are a time series of one optimisation run; row order is decision order. There are no relationships across functions.

**Recommended splits.** None for the optimisation task, since every point informs the next query. For validating a surrogate I use leave-one-out cross-validation within a function, never a random split across rounds, because later points depend on earlier ones.

**Privacy and people.** The data set contains no people, no personal data and no sub-populations.

## 3. Collection process

**How was the data acquired?** By submitting one query string per function per week to the course's capstone portal and recording the scalar it returned later that week. The functions are synthetic and their form is hidden; the only instrument is the portal.

**Sampling strategy.** Deterministic and model-based, not random. Rounds 1 to 3: Gaussian process with expected improvement, on log-transformed outputs where skewed and with a fitted noise term where noisy. Rounds 4 onward: the same surrogate plus an SVM filter that fences off the region rated in the top third of results, a second candidate produced by repeating the last successful step, and structural hypotheses drawn from each function's description (dose balance for the drug function, combined sugar and milk load for the recipe). Roughly a quarter of queries were deliberately exploratory. Where the surface beat the forecast twice in the same direction, the model lost its vote on that function and I extrapolated the measured trend.

**Time frame.** Starting data supplied in Module 12 (July 2026). Rounds 1 to 10 submitted weekly from Module 12 to Module 21 (late July to 1 October 2026). The data set grows by eight rows a week until Module 24.

**Ethical review, consent.** Not applicable; no human subjects. The functions are described by the course as analogues of real tasks, but the numbers are synthetic.

## 4. Preprocessing and uses

**Transformations applied.** Raw outputs are stored unchanged. For modelling, functions 1 and 5 are fitted on log(y), which is monotone and so leaves the location of the maximum unchanged; the log values are computed on the fly and not stored. Inputs are used as submitted; no scaling is needed since all lie in [0, 1]. The raw data is therefore always preserved.

**Intended uses.** Fitting surrogates to choose the next query; auditing the decision history; comparing acquisition strategies on the same record; producing the final capstone report.

**Inappropriate uses.** Estimating the global maximum of any function from these points alone, since the sampling is biased toward regions already thought good. Pooling outputs across functions, since scales differ by hundreds of orders of magnitude. Treating a single reading of function 2 as exact. Generalising anything about the real-world processes the functions are named after.

## 5. Distribution and maintenance

**Where is it available?** In the public GitHub repository `github.com/vugarnv/Capstone`, under `stage2/`, with the analysis notebook beside it.

**Terms of use.** MIT licence for my own rows and code. The course-supplied starting points remain the property of the programme and are included for reproducibility only.

**Who maintains it?** I do. The history is appended after each round's results arrive and committed with the round number in the commit message, so any earlier state can be recovered from the git history. Errors found later are corrected in place with a note in the commit. Maintenance stops at the end of the programme; the final commit will be tagged `final`.
