# Model Card: BBO Capstone Optimiser

Following the simplified model card template from Mini-lesson 21.2 (Mitchell et al., Model Cards for Model Reporting). Final version, after round 13 (Module 25, October 2026).

## 1. Model overview

**Name.** BBO Capstone Optimiser: Gaussian process with expected improvement, SVM feasibility filter, decision committee.

**Version.** 3.1, final. Version 1 (rounds 1 to 3) was a plain GP with expected improvement, with a log transform on skewed outputs and a noise term on noisy ones. Version 2 (rounds 4 to 7) added the SVM filter, the repeat-last-step candidate and structural hypotheses from the function descriptions. Version 3 (rounds 8 to 13) added the multi-seed refit check, the margin test and the rule that a function which beats the forecast twice loses its model vote. Round 13 was a certification round.

**Developer.** Vugar Nasibov, Imperial College Business School ML and AI Professional Certificate. Contact via the GitHub repository.

**Licence.** MIT (see `LICENSE`).

**Description.** A sequential optimiser that chose one query per week for each of eight unknown functions. It fits a surrogate to all past evaluations, proposes the point with highest expected improvement, checks that proposal against a structural candidate and a stability test, and submits the survivor. It is not a single model but a decision procedure with a model inside it, and the record of which part made each decision is in the query log.

## 2. Intended use

**Primary task.** Maximising an expensive, gradient-free, low-dimensional (2 to 8) black-box function under a fixed budget of one evaluation per round, with delayed feedback.

**Target users.** Me, for the capstone; peers and facilitators reviewing the approach; anyone facing the same shape of problem (hyperparameter tuning, experimental design, process optimisation) who wants a documented worked example.

**Recommended use cases.** Budgets of tens of evaluations. Continuous inputs on a bounded box. Outputs that are deterministic or have modest, roughly constant noise. Settings where a wrong query costs a round but not a person.

**Not recommended for.** High-dimensional problems (above roughly 10 inputs) where a GP on this few points cannot fit. Categorical or mixed inputs. Heavy heteroscedastic noise. Any setting where the query itself has real-world consequences, since the exploration budget deliberately spends some queries on points expected to be poor.

## 3. Training data

**Sources.** The query data set documented in `docs/datasheet.md`: 175 course-supplied starting points plus 104 of my own over thirteen rounds, one row per evaluation.

**Size per function.** 23 to 53 points; see the datasheet table.

**Preprocessing.** Log transform of outputs for the skewed functions (1, 5 and 7); a fitted noise term for the noisy function (2); no input scaling. The SVM filter is trained on a binary label, top third of results against the rest, recomputed each round.

**Modalities.** Numeric vectors and scalars only.

## 4. Evaluation metrics

**Metrics used.** Best value found per function, and its improvement over the course-supplied starting best. Leave-one-out cross-validated R² of the surrogate per function, re-run every round. Cross-validated accuracy of the SVM filter. Predicted margin of the proposed point over the incumbent, used as a gate: proposals with thin or negative margin are discarded. Stability: whether a proposal survives refitting across several random seeds. Prediction error (forecast minus outcome) logged for every query, with two errors in the same direction removing the model's vote on that function.

**Final results.** All eight functions ended above their starting best. Five produced a new best in the final three rounds.

| Fn | Starting best | Round 1 | After round 10 | Final best (round) | How it ended |
|---|---|---|---|---|---|
| 1 | 7.7 × 10⁻¹⁶ | 3.2 × 10⁻²⁴¹ | 1.980 | 1.998 (12) | Invisible on a raw scale; once fitted on log y it became a 200-order-of-magnitude ridge. Searched by bracketing along measured lines; gains came from the trend, not the acquisition function. Round 13 repeat returned the identical value: deterministic. |
| 2 | 0.611 | 0.643 | 0.643 | 0.643 (1) | New best in round 1 and never beaten. From round 5 the fitted noise exceeded any predicted gain, so later queries certified the incumbent or probed elsewhere; the round 13 probe at (0.70, 0.96) returned 0.641, within noise. |
| 3 | −0.035 | −0.093 | −0.0135 | −0.0089 (11) | Below the starting best in round 1. Structural hypothesis (balance between the three doses) chose every new best. The round 13 repeat of the round 11 query returned −0.024, so the function is noisy and the best is partly luck. |
| 4 | −4.03 | −1.98 | 0.682 | 0.682 (4) | Large gains in rounds 1, 3 and 4, model-led; diminishing returns after. Round 13 repeat of the round 4 query returned the identical value: deterministic. |
| 5 | 1,089 | 1,389 | 8,662 | 8,662 (9) | The log transform turned an apparently signal-less surface into the largest gain of the project. In rounds 7 and 8 a probe along one axis, which three model families predicted would decline, returned higher values; the model lost its vote and the measured trend led to the corner of the box in round 9. Rounds 10 to 12 tested each face near the corner; all lower. Round 13 repeat: identical, deterministic. |
| 6 | −0.714 | −0.539 | −0.238 | −0.203 (12) | Stuck in one corner until round 6; model-led gains once conditioned on the full history, with new bests in rounds 11 and 12. Round 13 came back lower. |
| 7 | 1.365 | 1.762 | 2.571 | 3.162 (12) | Every query to round 10 held the first input at 0 because the surrogate rated the edge. Moving it to 0.1 (round 11) and 0.2 (round 12) produced the two largest late gains; 0.3 in round 13 came back lower, so the ridge had turned. |
| 8 | 9.598 | 9.864 | 9.972 | 9.983 (12) | No surrogate could predict it at 40 points, so the SVM filter fenced off the good region. From round 8 the GP validated and chose rounds 10 to 12, each a new best. Round 13 repeat: identical, deterministic. |

Of the 104 queries, the surrogate chose 47, a measured trend 20, deliberate exploration 19, a structural hypothesis 8, the SVM fence 5, and 5 were repeats. Of the 40 queries that produced a new best, the surrogate chose 16, a measured trend 14, the SVM fence 4, a structural hypothesis 4 and the log-fitted surrogate 2. No exploration query and no repeat produced a new best directly, which is the expected shape: exploration pays through what it tells the next round. The notebook reproduces both counts.

**Fairness or bias checks.** No human groups are involved. The relevant bias is sampling bias: queries cluster where the model already expects good values, so the surrogate's confidence outside those clusters is unearned. This was checked by tagging exploratory queries, refusing to trust flat regions with fewer than a handful of points, and from round 11 placing the exploration query in the least-sampled quadrant rather than at the model's highest-variance point.

## 5. Assumptions and limitations

**Assumptions.** That each function is smooth enough for a Matern 2.5 kernel to interpolate, which was untrue for function 1 without the log transform. That noise, where present, is roughly constant across the domain. That the maximum lies inside the box rather than on its boundary, which was false for function 5. That last round's cross-validation verdict on a surrogate still holds this round, which round 8 showed to be false for function 8 in the good direction and round 13 showed to be optimistic for function 3.

**Constraints.** One evaluation per function per week, results delayed until later in the week, no gradients, no batch queries, and no way to re-query the same point cheaply.

**Failure modes.** A confident surrogate mean in an unsampled region (the GP's version of a hallucination), which the multi-seed refit and margin test are designed to catch. Length-scales that flip with one added point on the higher-dimensional functions. Proposals stuck on the domain edge, which cost function 7 ten rounds. Declaring a function finished when the good region has simply not been found yet; functions 5 and 6 both did this to me. And the opposite: a single lucky reading on a noisy function (function 3, round 11) taken as a real best.

## 6. Ethical considerations and life cycle

**Transparency and reproducibility.** Every query carries a tag saying what decided it, the notebook re-derives every model from the history, and the history can be truncated to any round to replay the decision with the data available at the time. The parts that are not reproducible from the code are the structural hypotheses I drew from the function descriptions and the judgement calls when the model and the structural candidate disagreed; those are written up in the weekly reflections on the course platform and summarised in the Module 25 retrospective. The decision tags were reconstructed from those reflections in Module 25 rather than logged live; this is the one place where the record is a summary rather than a transcript.

**Adaptation.** The pipeline transfers to any bounded low-dimensional black box by changing the data folder and the dimension list. What does not transfer automatically is the per-function configuration (transform, noise term, filter on or off), which has to be re-derived from the new problem's early data.

**Monitoring plan.** Cross-validation re-run every round; any surrogate whose LOO R² fell was demoted to the committee vote rather than the sole decider. Best-so-far tracked per function so that a round with no gain was visible immediately. The challenge is complete; no further monitoring.

**Repository.** `github.com/vugarnv/Capstone`. Final version, Module 25, October 2026.
