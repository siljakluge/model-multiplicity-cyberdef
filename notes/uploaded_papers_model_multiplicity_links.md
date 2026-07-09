# Links from Uploaded ACD Papers to Model Multiplicity Experiments

## Uploaded Papers

- `CSR2026_Silja.pdf`: Robustness Evaluation of Autonomous Cyber Defense Agents under Adversarial Strategy Variation.
- `PaperICMCIS.pdf`: Inductive Graph-Based Multi-Agent Reinforcement Learning for Autonomous Cyber Defence.

## Core Takeaways

The ICMCIS paper studies graph-based MARL for autonomous cyber defense in CAGE Challenge 4 and compares it with an expert-informed heuristic baseline. The important methodological point for this project is the use of explainability as a diagnostic tool: reward decomposition and action-frequency analysis reveal that the learned policy achieves non-trivial performance but overuses connectivity-control actions and underuses monitoring/analysis actions.

The CSR paper extends this line by evaluating robustness under multiple attacker strategies. It argues that aggregate reward under one fixed attacker configuration is insufficient, and introduces a robustness score combining average performance, within-strategy variability, and cross-strategy sensitivity. The heuristic policy is more robust than the learned GNN policy, even though the learned policy remains meaningfully better than an inactive baseline.

## Transfer to Supervised IDS Model Multiplicity

The current `mmcyber` setup maps this idea from RL policies to supervised intrusion-detection classifiers:

- Multiple models trained on the same task can reach similar aggregate performance while making different decisions on individual samples.
- Decision disagreement is analogous to cross-policy behavioral variation in the ACD papers.
- SHAP disagreement is analogous to explanation/action-decomposition disagreement: models may agree on accuracy while relying on different evidence.
- Dataset subset variation can be treated as a lightweight proxy for environmental or attacker-distribution variation.

## Concrete Research Questions

1. Do neural IDS classifiers with similar accuracy disagree systematically on the same traffic samples?
2. Are high-disagreement samples concentrated in specific attack categories or feature regimes?
3. Do models that agree on predictions still show divergent SHAP explanations?
4. Does training subset variation induce stronger explanation multiplicity than seed variation alone?
5. Can a robustness-style score combine accuracy, prediction instability, and explanation instability?

## Suggested Robustness Score for This Project

For model `m`, use:

```text
R(m) = performance(m)
       - lambda_decision * decision_instability(m)
       - lambda_expl * explanation_instability(m)
```

Possible components:

- `performance(m)`: macro-F1 or balanced accuracy.
- `decision_instability(m)`: mean pairwise disagreement against the other trained models.
- `explanation_instability(m)`: mean SHAP cosine distance or `1 - top_k_jaccard` against the other models.

This mirrors the CSR robustness score but adapts it to model multiplicity rather than attacker-strategy variation.

## Next Implementation Ideas

- Add a `robustness-score` command that ranks models by performance minus decision and explanation instability.
- Add per-class disagreement plots, especially for rare attack classes.
- Add sample-level exports for "high confidence disagreement" cases where models strongly disagree despite confident predictions.
- Add seed-vs-subset decomposition: compare average disagreement within the same subset fraction versus across subset fractions.
- Add a short paper-outline file framing the project as "robustness-oriented evaluation of multiplicity in cyber classifiers."
