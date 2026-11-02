# Datasheet: BBO capstone functions 1 to 8

This datasheet follows the Module 25 datasheet template section by section. The template is written for one function at a time; I have answered every question for all eight functions in one document, with a table wherever the answer differs by function and prose where it does not. The data it describes is in `stage2/query_log.csv` (my 104 queries and the portal's results) and `stage2/initial_data/` (the course-supplied starting points). Final version, after round 13.

## Function overview

**1. Which function does this datasheet describe?** All eight: functions 1 to 8 of the Stage 2 black-box optimisation challenge. Each is treated separately below; no numbers are ever pooled across functions.

**2. What real-world scenario does this function simulate?**

| Fn | Scenario given in the course brief |
|---|---|
| 1 | Locating a radiation source from detector readings in a two-dimensional area |
| 2 | A noisy log-likelihood surface with several local optima |
| 3 | Adverse reactions to a combination of three drug compounds; the reaction is negated so higher is better |
| 4 | A warehouse-placement model with four hyperparameters |
| 5 | Yield of a chemical process controlled by four settings |
| 6 | A cake recipe scored by a penalty on five ingredient quantities; the penalty is negated so higher is better |
| 7 | A machine learning model with six hyperparameters |
| 8 | A machine learning model with eight hyperparameters, where the brief says a strong local maximum is an acceptable target |

**3. What is the dimensionality of the input?** 2D, 2D, 3D, 4D, 4D, 5D, 6D and 8D for functions 1 to 8. Every input lies in [0, 1]; the portal rejects 1.000000, so 0.999999 is the effective upper bound.

**4. How many initial data points were provided?** 10, 10, 15, 30, 20, 20, 30 and 40. The arrays are `initial_inputs.npy` of shape (n, d) and `initial_outputs.npy` of shape (n,).

**5. What does the output represent?**

| Fn | Output | Range in the starting data |
|---|---|---|
| 1 | Detector reading, a likelihood of the source being at the queried point | -0.0036 to 7.7e-16 |
| 2 | Log-likelihood at the queried parameters | -0.066 to 0.611 |
| 3 | Negated severity of the adverse reaction (ceiling 0) | -0.40 to -0.035 |
| 4 | Score of the placement model, higher is better | -32.6 to -4.03 |
| 5 | Yield of the process | 0.11 to 1,089 |
| 6 | Negated recipe penalty (ceiling 0) | -2.57 to -0.71 |
| 7 | Performance score of the model | 0.0027 to 1.365 |
| 8 | Performance score of the model | 5.59 to 9.598 |

## Nature of the data

**1. Describe the structure of the initial dataset.** For each function, an input matrix of shape (n, d) with n and d as above, and an output vector of shape (n,). No labels, no missing values, no metadata about how the starting points were chosen; they look space-filling rather than concentrated.

**2. How does the dataset evolve as you add new queries weekly?** One point per function per week, thirteen in total, so each function grew from its starting n to n + 13 (23, 23, 28, 43, 33, 33, 43 and 53 points). The sampling pattern shifted over the weeks. Rounds 1 to 3 followed the GP's expected improvement, which spread queries widely because the model was uncertain everywhere. From round 4 most queries landed near the best known point, and I reserved specific rounds for deliberate exploration: six exploratory queries on function 2 (its surface was closed early), three each on functions 5, 7 and 8, two on function 3, one on function 6, none on functions 1 and 4. Queries on the domain edge (a coordinate at 0.000000 or 0.999999) number 0, 1, 4, 0, 13, 9, 10 and 7 for functions 1 to 8; function 5's are the corner it converged to, function 7's are the first input held at zero until round 10, function 6's the milk quantity at zero.

**3. Does the function include noise or randomness?** I tested this directly in round 13 by resubmitting the best known input on five functions.

| Fn | Evidence | Verdict | Effect on strategy |
|---|---|---|---|
| 1 | Repeat returned 1.997940, identical to 15 significant figures | Deterministic | None needed |
| 2 | Inputs 0.005 apart in x1 returned 0.51 and 0.64; a GP noise term fitted on all 23 points estimates a standard deviation near 0.05 | Noisy | Differences under about 0.05 were treated as chance; late queries were fresh points on the same crest rather than attempts to resolve it |
| 3 | Repeat of the round 11 input returned -0.0244 against -0.0089 the first time | Noisy, more than the early fits suggested | The round 11 best is partly luck; I stopped treating small gains on this function as real |
| 4 | Repeat returned 0.681621, identical | Deterministic | None needed |
| 5 | Repeat returned 8662.405, identical | Deterministic | Confirmed the corner as the final answer |
| 6 | Not repeated | Unknown; the surface is rough enough that noise and ruggedness cannot be separated from my data | Treated as deterministic |
| 7 | Not repeated | Unknown | Treated as deterministic |
| 8 | Repeat returned 9.98279, identical | Deterministic | None needed |

**4. Based on observations, does the function appear unimodal, multimodal, noisy or smooth?**

| Fn | What the data shows | Reasoning |
|---|---|---|
| 1 | One sharp peak near (0.629, 0.628) with a ring of negative readings around it (for example -0.0036 at (0.650, 0.682)) | Only visible on the log scale, where the readings span more than 200 orders of magnitude and form a single bowl; no second rise anywhere in the 23 points |
| 2 | A narrow crest at x1 = 0.700, about 0.005 wide, flat along x2 from 0.17 to 0.96; a lower shoulder near x1 = 0.45 (readings 0.21 to 0.25); dead elsewhere | The fitted GP length-scale in x1 is 0.03 and in x2 is at its upper bound; this matches the brief's "local optima" |
| 3 | A plateau near the ceiling around (0.48 to 0.58, 0.57 to 0.66, 0.39 to 0.44); the third input is sharp and costly above about 0.45; the first input flat | GP length-scale 0.19 on x3 against about 1 on the others; the random forest was the only model with positive cross-validated R2 on the final data, which suggests steps rather than smooth curvature; noisy as above |
| 4 | Rugged. One spike of 0.68 at (0.376, 0.415, 0.419, 0.417); points 0.01 to 0.03 away return 0.40 to 0.63; points 0.08 away return -1.4 to -2.3 | No smooth model predicted the neighbourhood well: GP forecasts of 0.85 and 0.71 returned 0.40 and 0.63. Multimodal in the sense that the starting data had its best in a different region |
| 5 | Smooth and monotone towards the corner (1, 1, 1, 1), where the maximum is. Symmetric in x1, x3 and x4 near the corner: lowering any one of them to 0.9 returns the identical 7073.39 | Steep: 0.9 on one axis costs 18 percent; the brief said unimodal, and nothing in 33 points contradicts it |
| 6 | One basin found around (0.47, 0.33, 0.60, 0.75, 0.00); a peer reported a value of -0.10 against my -0.20, so a better basin exists that my queries never reached | Multimodal by that evidence. Two ingredients (butter, the fourth input, and milk, the fifth) carry most of the random-forest importance |
| 7 | A ridge rising as the first input moves from 0 to about 0.25, then turning: 0.1 gave 2.88, 0.2 gave 3.16, 0.3 gave 3.10. The other inputs have broad optima around 0.14, 0.33, 0.30, 0.31 and 0.67 | Smooth; GP cross-validated R2 of 0.75 on log(y) at 43 points |
| 8 | Smooth, with a near-linear backbone: a linear SVM separates the top third of results from the rest at about 90 percent cross-validated accuracy on the final data (95 percent when first fitted at 40 points), and a PCA of the top third puts over 80 percent of its spread in three components | GP cross-validated R2 of 0.98 at 53 points; the best region is narrow in a few directions and broad in the others |

## Your optimisation strategy

**1. Which optimisation method(s) did you use?** The core was Bayesian optimisation: a Gaussian process surrogate with a Matern 2.5 kernel and per-dimension length-scales, and expected improvement as the acquisition function, maximised by multi-start L-BFGS-B. Around it, by function:

| Fn | Methods used |
|---|---|
| 1 | GP + EI on log(y) (rounds 2, 4, 5); quadratic triangulation of the source from log readings (round 3); measured bracketing and bisection along lines through the best point, with a parabola through three readings for the final steps (rounds 6 to 12) |
| 2 | GP + EI with a fitted noise term (rounds 1, 2); crest mapping by hand once the GP had nothing to add; deliberate probes of the unmapped corners |
| 3 | GP + EI with a noise term; a structural hypothesis (balanced doses, round 3); the model committee's local step when the GP's cross-validation went negative (rounds 7, 9, 12); the GP's local mean maximum within a small box (rounds 10, 11); two edge probes |
| 4 | GP + EI; a neural-network gradient estimate tested head to head against the GP (round 5, the GP won); the committee's choice when the GP over-smoothed the spike; a sharper GP with tightened length-scale bounds for the last local steps |
| 5 | GP + EI on log(y) (rounds 1 to 6); a neural-network gradient hypothesis (round 7); following the measured ridge along x1 to the corner (rounds 8, 9); edge probes to map the corner (rounds 10 to 12) |
| 6 | GP + EI; two structural hypotheses from ingredient correlations (rounds 2, 3, both lost); a single-axis test of the butter quantity (round 8); committee refinement (round 9); a test of a peer-reported recipe (round 10); GP local maximum (rounds 11 to 13) |
| 7 | GP + EI on log(y); three designed explorations of the high-x2, high-x3 region (rounds 3 to 5, all low); moving the first input off zero by hand (round 11); the GP's local maximum (round 12); a further step along x1 (round 13) |
| 8 | GP + EI inside a linear SVM fence that separates the top third of results (rounds 3, 4, 6, 7, 9); plain GP + EI once it validated at 0.9 cross-validated R2 (rounds 10 to 12); two far probes |

**2. Why did you choose this method for this particular function?** Expected improvement is the standard answer when evaluations are expensive and few, which is every function here. The per-function additions follow from the diagnostics: the log transform where outputs span orders of magnitude (1, 5, 7), the noise term where readings at nearby inputs disagree (2, 3), the SVM fence where eight dimensions and forty points left regression with nothing to say but classification with plenty (8), the committee where the GP's cross-validation went negative (3, 4, 6), and measured bracketing where quadratic fits had misled me (1).

**3. How did you balance exploration and exploitation?** Expected improvement does most of the balancing by rewarding both high predicted value and high uncertainty. On top of that, 17 of the 104 queries were deliberate explorations, placed where the brief warned of multiple optima (function 2) or where a region was unsampled (functions 3, 5, 7, 8). The weighting moved with the calendar: exploration in the early and middle rounds, when a discovery could still be exploited, and refinement only in rounds 12 and 13. The rule I used from round 4 was that a proposal is submitted automatically only if the GP predicts a clear gain over the best point; otherwise that function gets a deliberate decision, which is where the exploration queries came from.

**4. Did your strategy change over the weeks? Why?** Three times. After round 4 I audited every query so far: the model's proposals had improved the best point in 9 of 13 cases and my own hypotheses in 3 of 9, so the model became the default and my role became checking it. From round 8 I added two gates before submitting: the proposal had to be stable across five random seeds, and its predicted gain had to be clearly positive. From round 9 I added the model committee on stalled functions and a calibration record of forecast against outcome per function, and let the record decide how far to trust each model. The final rounds were refinement, then certification by repeating the best inputs.

## Data handling and preprocessing

**1. Did you rescale or normalise inputs? Why or why not?** No. Every input already lies in [0, 1] on the same scale, so the Matern kernel's per-dimension length-scales do the work that scaling would otherwise do.

**2. Did you train any surrogate models?** Yes: a Gaussian process per function (the main surrogate); a linear SVM classifier on function 8; and a committee of ridge regression, k-nearest neighbours, random forest, gradient boosting and a small neural network, compared by five-fold cross-validation, on the functions where the GP had stalled. A quadratic surface fitted to log|y| served as a one-off source estimator on function 1.

**3. If yes, what preprocessing did the surrogate require?** Outputs standardised inside the GP (scikit-learn's `normalize_y`). A log transform of outputs for functions 1, 5 and 7, with function 1's single negative starting value floored at 1e-300 first. A fitted white-noise term in the kernel for functions 2 and 3. Length-scale bounds of 0.02 to 5 on function 4 after the default bounds let the GP smooth away the spike. The kernel optimiser restarted from several initial values and the acquisition optimiser from 30 to 40 random starts; a proposal was accepted only if it did not move when the random seed changed.

**4. Did you handle outliers or unusual data points?** Nothing was removed. Function 4's large negative readings far from the good region are real and were kept; the problem they cause is that standardisation lets them dominate the fit, which is why the local steps used the sharper length-scale bounds and the committee check. Function 1's negative starting value was floored before the log transform rather than dropped, since it carries the information that the detector reads negative in a ring around the source. On the noisy functions the fitted noise term, not any filtering, decided which differences were meaningful.

## Weekly iteration and learning

**1. How did new data points change your understanding of the function landscape?** Function 1 changed in round 2, when the log transform turned a flat surface into a gradient and the next query jumped eleven orders of magnitude. Function 5 changed in rounds 7 to 9, when readings along x1 rose where three model families had predicted a fall, and the maximum turned out to be the corner of the box rather than the interior point the models favoured. Function 7 changed in round 11, when the first input, held at zero by every model proposal since round 1, moved to 0.1 and produced the largest single gain of the challenge. Function 2 changed in rounds 11 and 12, when the refitted noise term came out ten times larger than my early estimate and told me the crest readings were draws from one distribution. Function 3 changed in round 13, when the repeat showed the same.

**2. Did you encounter local optima? How did you detect them?** On function 2 the shoulder near x1 = 0.45 is a second, lower rise; I detected it from two starting-data readings higher than their neighbours and left it alone once the crest at 0.70 was mapped. On function 4 the starting data's best region and the region found in round 3 were different, and the spike found in round 4 sits in a neighbourhood where points 0.03 away are far lower; I detected the ruggedness from repeated forecast misses. On function 6 the better basin reported by a peer is one my queries never entered; the committee and GP both placed their optimum inside the basin I had, which is the signature of a surrogate that cannot see beyond its data.

**3. Which queried inputs were most informative and why?** Function 1 round 2, the first log-scale step, because it proved the surface had structure. Function 5 rounds 7 and 8, the ridge readings that contradicted every model, because they located the true maximum. Function 7 round 11, the first move off the x1 = 0 edge, because it showed that an unmeasured direction is not an unimportant one. Function 8 round 3, the first SVM-fenced query, because it showed classification could guide the search where regression could not. Function 3 round 13 and function 2 rounds 9 to 12, the noise readings, because they changed what a best value means on those functions.

**4. If you restarted, what would you do differently?** Look at every function on a log scale before fitting anything. Move each coordinate of the best point at least once in the first five rounds, so that no axis stays unmeasured until round 11. Spend one early repeat on any function the brief calls noisy, so the noise level is known before it matters. Run the committee from round 1 rather than round 9. And in the final round, under a scoring rule that counts only the last submission, resubmit every best point: I took three model-backed steps in round 13 (functions 2, 6 and 7) and all three came back below the best.

## Performance and results

**1. What is the best output value you achieved?** and **2. Which input vector produced this value?**

| Fn | Starting best | Best value found | Round | Input that produced it | Final submission (round 13) |
|---|---|---|---|---|---|
| 1 | 7.7e-16 | 1.99794 | 12 | 0.629100-0.628200 | 1.99794 (same input) |
| 2 | 0.611 | 0.64296 | 1 | 0.700000-0.500000 | 0.64071 at 0.700000-0.960000 |
| 3 | -0.0348 | -0.00894 | 11 | 0.483000-0.571000-0.391600 | -0.02441 (same input, noisy) |
| 4 | -4.03 | 0.68162 | 4 | 0.376347-0.414508-0.418580-0.417207 | 0.68162 (same input) |
| 5 | 1,089 | 8662.41 | 9 | 0.999999-0.999999-0.999999-0.999999 | 8662.41 (same input) |
| 6 | -0.714 | -0.20342 | 12 | 0.470100-0.325000-0.604400-0.752400-0.000000 | -0.34314 at a GP step |
| 7 | 1.365 | 3.16217 | 12 | 0.200000-0.143100-0.333100-0.301400-0.308800-0.673000 | 3.09807 at x1 = 0.3 |
| 8 | 9.598 | 9.98279 | 12 | 0.084300-0.195900-0.140000-0.113700-0.950000-0.504600-0.228600-0.629000 | 9.98279 (same input) |

**3. How confident are you that this is near the global maximum? Why?**

| Fn | Confidence | Reason |
|---|---|---|
| 1 | High | Peak bracketed on both axes to within 0.002; a parabola through the last readings predicts no more than 0.001 further gain; the log-scale GP rates every other region hundreds of orders of magnitude lower |
| 2 | Moderate | The crest is fully mapped and flat; with noise near 0.05 the recorded best is partly a lucky draw, and peers' readings of 0.68 on the same crest are consistent with that |
| 3 | Moderate | The ceiling is 0 and the plateau sits at about -0.01 to -0.02; the noise is of the same size as any remaining gain |
| 4 | Low to moderate | The spike is real and repeatable, but the neighbourhood is rugged and no model predicts it, so a higher spike within 0.05 cannot be excluded |
| 5 | High | All four inputs at the upper bound, the function monotone on every axis tested, deterministic, and the same corner value reported by peers |
| 6 | Low | A peer's -0.10 shows a better basin exists; my surrogate never pointed outside the basin it had |
| 7 | Moderate | The ridge along x1 turned between 0.2 and 0.3, the other axes have been scanned, and the GP's forecast of the turn was right; a peer's 3.25 suggests a little more is available nearby |
| 8 | Moderate to high | Three consecutive GP forecasts within 0.003 of the outcome, cross-validated R2 of 0.98, and the brief accepts a strong local maximum |

**4. Did your results align with expectations for this function?** Mostly. Function 1 behaved like a source with a decaying signal, including the negative ring the brief did not mention. Function 2 showed exactly the local optima and noise promised. Function 5 was unimodal as stated, but the maximum on the boundary rather than inside surprised me and cost two rounds. Function 4 was rougher than a "hyperparameter" surface is usually described. Functions 7 and 8 behaved as tuning surfaces usually do, broad optima in most inputs and a few that matter.

## Ethical, practical and general considerations

**1. How does this black-box optimisation task relate to real-world applications?** It is the same problem as tuning a model's hyperparameters, choosing settings for a chemical or manufacturing process, or designing a sequence of experiments: an unknown objective, expensive evaluations, no gradients, a small budget. My own interest is in venture investing, where the equivalent question is which candidate gets the next expensive piece of attention, and the transferable habit is making each costly evaluation answer a question written down in advance.

**2. What limitations arise from the synthetic nature of the function?** The functions are deterministic or have simple noise, return instantly once a week, and never fail or return an error. Real objectives drift over time, have heteroscedastic noise, fail for some inputs, and cost different amounts in different regions. The brief's stories (radiation, drugs, cake) are labels, and nothing in my results says anything about those real processes. The weekly cadence also removed the batch question that real experiments face: here one query per function was the rule, not a choice.

**3. Would your strategy scale to more serious or more expensive problems? Why or why not?** To a point. The GP with expected improvement is the standard tool for budgets of tens to a few hundred evaluations and up to roughly ten inputs, which covers hyperparameter tuning and small experimental designs. Above that, the GP's cost and the curse of dimensionality bite; I would move to a trust-region method and a library built for it. The parts of my process that scale best are not the model but the habits: the log-scale check, the calibration record of forecast against outcome, and the rule that a model keeps its authority only while its forecasts hold up. The part that scales worst is the hand-mapping I did on functions 1 and 2, which only works in two dimensions.

**4. What risks or pitfalls should a future user be aware of when analysing this function?** For function 1, fit on the log scale or you will see nothing, and do not trust a quadratic fit that includes points far from the peak. For function 2, do not read differences under 0.05 as signal, and do not assume the GP's smooth length-scale along x2 means x2 is irrelevant everywhere. For function 3, repeat before you believe a best value. For function 4, no smooth surrogate predicts the neighbourhood of the spike; use several model families and small steps. For function 5, the maximum is on the boundary, so check the corners before refining the interior. For function 6, the surrogate's optimum stays inside the basin you have sampled; budget explicit exploration. For function 7, an input that every proposal leaves at its bound has usually not been tested, not found unimportant. For function 8, the SVM fence depends on the top-third threshold; check that the fence still has candidates inside it after each new round.
