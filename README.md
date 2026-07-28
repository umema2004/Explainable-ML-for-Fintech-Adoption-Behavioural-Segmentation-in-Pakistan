# Explainable ML for Fintech Adoption and Behavioural Segmentation in Pakistan

Using explainable machine learning (Random Forest / XGBoost + SHAP) and unsupervised segmentation (k-means), this project identifies who is excluded from Pakistan's digital financial system — and, more importantly, where formal access doesn't translate into actual usage. The analysis is backed by a full robustness, calibration, and fairness validation suite, and is written up as an IEEE-format conference paper.

## Headline finding

**44.3% of the sampled population has near-universal mobile phone ownership (98%) and majority internet use (62.3%), yet two-thirds of this group remain formally excluded from the financial system.** This pattern was independently confirmed through three separate analytical routes:
1. **Unsupervised clustering** — k-means found this segment without being told about any predefined adoption categories.
2. **Ablation study** — removing behavioural features costs far more predictive power (macro F1 0.494 → 0.415) than removing access features, which actually *improves* performance slightly (→ 0.506).
3. **Model error analysis** — the demographic profile of people the classifier mistakenly predicts as "digitally engaged" (93.5% mobile phone ownership, 80.6% workforce participation) closely matches this same access-rich, adoption-poor cluster.

A second major finding: **gender is the single strongest structural predictor in the entire analysis** (Cramer's V = 0.807 for cluster membership, exceeding urban/rural status by a wide margin). The most digitally-excluded cluster is 96.3% female, and gender survives as a top-4 SHAP driver even after controlling for age, income, and workforce status.

## Data

- **Source:** [Global Findex Database 2025, Pakistan microdata](https://microdata.worldbank.org/index.php/catalog/7961) (Ref: `PAK_2024_FINDEX_v02_M`), World Bank Microdata Library.
- **Coverage:** 1,000 individual-level respondents, fieldwork May–Dec 2024, 183 raw variables.
- **Not included:** raw microdata file is not committed to this repo. Download it yourself from the link above and place it in the project root before running the pipeline.

## Method overview

1. **Feature selection** — 20 variables across four construct groups (demographics, access, behaviour, resilience), mapped from the Findex codebook.
2. **Target construction** — a 4-level ordinal **Adoption Tier** (Excluded → Basic → Digitally Engaged → Advanced).
3. **Supervised modelling** — Logistic Regression → Decision Tree → Random Forest → XGBoost, with `account`, `dig_account`, `anydigpayment`, `saved`, `fin22a`, `account_fin`, `account_mob` explicitly excluded from the feature set (an early version leaked these into the model and produced a spurious ~100% accuracy — caught and corrected; see script 02 comments).
4. **Hyperparameter tuning** — `RandomizedSearchCV`, 25 iterations, 5-fold stratified CV, macro F1 scoring, for both Random Forest and XGBoost.
5. **Robustness analysis** — repeated stratified 5-fold CV (10 repeats, 50 folds total) to confirm performance isn't an artifact of one train/test split.
6. **Explainability** — SHAP TreeExplainer (multiclass-averaged) plus dependence plots for the top 4 features against the Advanced-tier class.
7. **Calibration analysis** — per-class reliability diagrams and Brier scores.
8. **Feature importance consistency** — Gini vs. permutation vs. SHAP, compared via Spearman rank correlation.
9. **Error analysis** — confusion pairs and subgroup accuracy (gender, urban/rural), including a documented accuracy-paradox caveat (see Ethics, below).
10. **Ablation study** — feature-group leave-one-out and single-group-only configurations.
11. **Unsupervised segmentation** — k-means (k=2–5), k selected via silhouette score (Davies-Bouldin and Calinski-Harabasz reported alongside).
12. **Cluster validation** — bootstrap stability (100 resamples, Adjusted Rand Index), an algorithm-independent cross-check against Ward agglomerative clustering, and an explicit k=2 vs. k=3 interpretability comparison.
13. **Statistical validation** — chi-square tests of independence with Cramer's V, Wilson score 95% confidence intervals on all headline proportions.
14. **Literature comparison** — results compared directly against Chowdhury et al. (2026, Bangladesh), Emuron et al. (2025, South Africa), Saidy & Hassan (2025, Sub-Saharan Africa), and Suri & Jack (2016, Kenya).

## Results at a glance

| Model | Accuracy | F1 (macro) |
|---|---|---|
| XGBoost | 0.800 | 0.481 |
| Random Forest (default) | 0.708 | 0.486 |
| **Random Forest (tuned, model of record)** | — | **0.494** |
| Decision Tree | 0.488 | 0.424 |
| Logistic Regression | 0.420 | 0.392 |

Repeated CV (50 folds): mean macro F1 = 0.480 (95% CI: 0.447–0.521) — confirms the tuned estimate is stable.

| Cluster | Size | Profile | Mean Adoption Tier | % Excluded |
|---|---|---|---|---|
| 0 — Digitally Excluded Women | 43.1% | 96.3% female, 51% rural, 9.7% internet use | 0.21 | 89.3% |
| 1 — Connected but Unengaged | 44.3% | 16.3% female, 98% mobile phone, 62.3% internet use | 0.80 | 66.8% |
| 2 — Advanced Adopters | 12.6% | Highest income/education, 84–98% on all behaviour flags | 2.92 | 0.8% |

Cluster stability: bootstrap ARI = 0.904 (95% CI: 0.847–0.982); ARI vs. Ward agglomerative clustering = 0.740.

## Repository structure

```
.
├── Findex_Microdata_2025_Pakistan.csv       # (not committed - download separately)
├── hypothesis_variable_selection.md          # Week 1: methods & variable mapping
├── 01_load_and_construct_target.py           # data loading, cleaning, target construction
├── 02_model_comparison.py                    # supervised modelling + SHAP
├── 03_segmentation.py                        # k-means segmentation + PCA
├── 04_statistical_tests.py                   # chi-square tests + confidence intervals
├── 05_model_diagnostics.py                   # tuning, robustness, calibration, ablation, error analysis
├── 06_cluster_validation.py                  # bootstrap stability, algorithm cross-check, k=2 vs k=3
├── fintech_adoption_pakistan.tex              # IEEE-format paper (single file, inline bibliography)
├── project_status_summary.md                 # running project log
└── README.md
```

## How to run

Requires Python 3.9+.

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy statsmodels
```

1. Download the Findex Pakistan 2024 CSV from the [World Bank Microdata Library](https://microdata.worldbank.org/index.php/catalog/7961) and place it in the project root.
2. Run the pipeline in order:

```bash
python 01_load_and_construct_target.py
python 02_model_comparison.py
python 03_segmentation.py
python 04_statistical_tests.py
python 05_model_diagnostics.py      # requires 02's output
python 06_cluster_validation.py     # requires 03's output
```

Each script reads the output of an earlier script and saves its own CSV/PNG outputs to the project root. Scripts 05 and 06 are diagnostic/validation layers and can be skipped for a faster run-through, but their outputs are referenced throughout the paper.

## Building the paper

`fintech_adoption_pakistan.tex` is a self-contained IEEE conference-format LaTeX file (uses `IEEEtran.cls`) with the bibliography inline (no separate `.bib` file). Figure references use placeholder filenames matching the PNGs produced by scripts 01–06 — copy those PNGs into the same folder as the `.tex` file, then compile:

```bash
pdflatex fintech_adoption_pakistan.tex
pdflatex fintech_adoption_pakistan.tex   # second pass resolves citations/references
```

## Hypotheses tested

- **H1 (access-usage gap):** Confirmed and triangulated three independent ways — direct dormant-account rate (6.6%), unsupervised clustering (44.3% of sample), and model error profiling.
- **H2 (rural vs. urban):** Urban adoption exceeds rural adoption (χ² = 21.742, p = 0.0001), consistent with mainstream inclusion literature rather than Chowdhury et al.'s reversed Bangladesh finding.
- **H3 (drivers):** Behavioural variables and gender outrank pure access variables as predictors — confirmed independently via SHAP, ablation, and Gini importance (though permutation importance disagrees more substantially; see Limitations).

## Ethical considerations

- **Fairness / accuracy paradox:** Raw subgroup accuracy (women 0.852 vs. men 0.689) is misleading here — gender is so strongly entangled with outcome (Cramer's V = 0.807) that it mechanically inflates accuracy for the subgroup concentrated in the largest, easiest-to-predict class. Subgroup-level macro F1 or per-class recall is recommended over aggregate accuracy for fairness audits in this kind of setting.
- **Risk of misuse:** This model is intended for aggregate, population-level policy targeting (e.g., identifying segments for trust-building interventions), not individual-level credit or loan-eligibility decisions, which would risk encoding existing gender and rural disparities into automated lending.
- **Calibration caution:** The Excluded class is poorly calibrated (systematically underconfident); predicted probabilities should not be used in downstream resource-allocation formulas without recalibration.
- **Data privacy:** All analysis uses de-identified, publicly available World Bank microdata collected under Findex's standard informed-consent protocols.

## Limitations

- Sample size (n=1,000) limits reliability for small subgroups — the Basic tier (n=21) is essentially unpredictable by any model tested.
- `fin17b` (saved via mobile money), the top-ranked predictor by Gini and SHAP, is partly confounded with mobile money account ownership due to Findex's survey routing logic (~74% of respondents routed out of this question).
- Permutation importance disagrees substantially with Gini/SHAP for several features (including negative importance for gender and rural status) — likely a correlated-feature attribution effect rather than evidence these features are unimportant, but a genuine limitation of relying on any single importance method.
- The k=3 cluster solution's silhouette score is only marginally better than k=2 — validated independently via bootstrap stability and algorithm cross-check, but not an overwhelmingly dominant structure by silhouette score alone.
- Calibration is poor for three of four tiers; only the Advanced class is well-calibrated.
- All language is associational, not causal; Findex is a cross-sectional survey.

## Data citation

World Bank. 2025. *Global Findex Database 2025, Pakistan.* Ref: PAK_2024_FINDEX_v02_M. World Bank Microdata Library. https://microdata.worldbank.org/index.php/catalog/7961
