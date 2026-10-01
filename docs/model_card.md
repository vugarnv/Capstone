# Model Card: BBO Capstone Optimiser

Following the simplified model card template from Mini-lesson 21.2 (Mitchell et al., Model Cards for Model Reporting). Last updated after round 10 (Module 21, October 2026).

## 1. Model overview

**Name.** BBO Capstone Optimiser: Gaussian process with expected improvement, SVM feasibility filter, decision committee.

**Version.** 3.0, round 10. Version 1 (rounds 1 to 3) was a plain GP with expected improvement. Version 2 (rounds 4 to 7) added the SVM filter, the repeat-last-step candidate and the log and noise transforms. Version 3 (rounds 8 onward) added the multi-seed refit check, the margin test and the rule that a function which beats the forecast twice loses its model vote.

**Developer.** Vugar Nasibov, Imperial College Business School ML and AI Professional Certificate. Contact via the GitHub repository.

**Licence.** MIT.

**Description.** A sequential optimiser that chooses one query per week for each of eight unknown functions. It fits a surrogate to all past evaluations, proposes the point with highest expected improvement, checks that proposal against a structural candidate and a stability test, and submits the survivor. It is not a single model but a decision procedure with a model inside it.

## 2. Intended use

**Primary task.** Maximising an expensive, gradient-free, low-dimensional (2 to 8) black-box function under a fixed budget of one evaluation per round, with delayed feedback.

**Target users.** Me, for the capstone; peers and facilitators reviewing the approach; anyone facing the same shape of problem (hyperparameter tuning, experimental design, process optimisation) who wants a documented worked example.

**Recommended use cases.** Budgets of tens of evaluations. Continuous inputs on a bounded box. Outputs that are deterministic or have modest, roughly constant noise. Settings where a wrong query costs a round but not a person.

**Not recommended for.** High-dimensional problems (above roughly 10 inputs) where a GP on this few points cannot fit. Categorical or mixed inputs. Heavy heteroscedastic noise. Any setting where the query itself has real-world consequences, since the exploration budget deliberately spends some queries on points expected to be poor.

## 3. Training data

**Sources.** The query data set documented in `docs/datasheet.md`: 175 course-supplied starting points plus 80 of my own after ten rounds, one row per evaluation.

**Size per function.** 20 to 50 points; see the datasheet table.

**Preprocessing.** Log transform of outputs for the two skewed functions (1 and 5); a fitted noise term for the noisy function (2); no input scaling. The SVM filter is trained on a binary label, top third of results against the rest, recomputed each round.

**Modalities.** Numeric vectors and scalars only.

## 4. Evaluation metrics

**Metrics used.** Best value found per function, and its improvement over the course-supplied starting best. Leave-one-out cross-validated R² of the surrogate per function, re-run every round. Cross-validated accuracy of the SVM filter. Predicted margin of the proposed point over the incumbent, used as a gate: proposals with thin or negative margin are discarded. Stability: whether a proposal survives refitting across several random seeds.

**Results after ten rounds.** The best-so-far value for every function is held in the query history (`stage2/`) and updated after each round. The table below gives the starting best from the course data, the first result my pipeline produced in round 1, and where each function stands after ten rounds.

| Fn | Starting best | Round 1 result | Status after round 10 |
|---|---|---|---|
| 1 | 7.7 × 10⁻¹⁶ | 3.2 × 10⁻²⁴¹ | Treated as uninformative in round 1, then recognised as a 200-order-of-magnitude gradient once fitted on log scale. Searched by bracketing and bisection along measured lines; gains came from that, not from acquisition. |
| 2 | 0.611 | 0.643 | New best in round 1. Saturated at about sixteen points; the fitted noise level now exceeds any predicted gain, so recent queries certify the incumbent rather than chase improvement. |
| 3 | −0.035 | −0.093 | Ceiling-at-zero function. Structural hypothesis (balance between the three doses) predicts the outcome better than any single compound. Diminishing returns; budget now spent certifying. |
| 4 | −4.03 | −1.98 | Large gain in round 1 (halved the distance to zero). Diminishing returns since; one of the functions where fitted noise is larger than predicted gain. |
| 5 | 1,089 | 1,389 | New best in round 1. The log transform turned an apparently signal-less surface into my largest single gain. In rounds 8 to 9 a probe along one axis, which three model families predicted would decline, returned a higher value, and a longer stride the next week gained eleven per cent. The model has lost its vote here; I extrapolate the measured ridge. |
| 6 | −0.714 | −0.539 | New best in round 1. Structural hypothesis (combined sugar and milk load) outperforms single-ingredient effects. Stuck in one corner until round 6, when reasoning across past points broke the loop. |
| 7 | 1.365 | 1.762 | New best in round 1. Under-performed in weeks 1 to 3 while run zero-shot from raw acquisition; improved once conditioned on the full history. |
| 8 | 9.598 | 9.864 | New best in round 1. No surrogate could predict it at 40 points, so the SVM filter (95% cross-validated accuracy) fenced off the good region. At 48 points the GP suddenly validates at LOO R² 0.95, the best fit anywhere; round 10 is its first model-led query. |

Seven of eight functions improved on the course-supplied best in round 1; function 1 did not until the log-scale fix. Full per-round values and the decision tag for every query are in the history.

**Fairness or bias checks.** No human groups are involved. The relevant bias is sampling bias: queries cluster where the model already expects good values, so the surrogate's confidence outside those clusters is unearned. This is checked by tagging exploratory queries and by refusing to trust flat regions that have fewer than a handful of points.

## 5. Assumptions and limitations

**Assumptions.** That each function is smooth enough for a Matern-2.5 kernel to interpolate, which was untrue for function 1 without the log transform and may be untrue elsewhere. That noise, where present, is roughly constant across the domain. That the maximum lies inside the box rather than on its boundary. That last round's cross-validation verdict on a surrogate still holds this round, which round 8 showed to be false for function 8.

**Constraints.** One evaluation per function per week, results delayed until later in the week, no gradients, no batch queries, and no way to re-query the same point cheaply.

**Failure modes.** A confident surrogate mean in an unsampled region (the GP's version of a hallucination), which the multi-seed refit and margin test are designed to catch. Length-scales that flip with one added point on the higher-dimensional functions. Proposals stuck on the domain edge. Declaring a function finished when the good region has simply not been found yet; functions 5 and 6 both did this to me.

## 6. Ethical considerations and life cycle

**Transparency and reproducibility.** Every query carries a tag saying what decided it, the notebook re-derives every model from the history, and each round is a separate commit, so another researcher can replay any round with the data available at that time and check whether they would have chosen the same point. The parts that are not reproducible from the code are the structural hypotheses I drew from the function descriptions and the judgement calls when the model and the structural candidate disagreed; those are written up in the weekly reflections.

**Adaptation.** The pipeline transfers to any bounded low-dimensional black box by changing the data file and the dimension list. What does not transfer automatically is the per-function configuration (transform, noise term, filter on or off), which has to be re-derived from the new problem's early data.

**Monitoring plan.** Cross-validation re-run every round; any surrogate whose LOO R² falls is demoted to the committee vote rather than the sole decider. Best-so-far tracked per function so that a round with no gain is visible immediately.

**Repository.** `github.com/vugarnv/Capstone`. Last updated: Module 21, round 10.
