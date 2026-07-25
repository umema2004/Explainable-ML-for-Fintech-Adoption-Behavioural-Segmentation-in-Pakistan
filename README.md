# Explainable ML for Fintech Adoption & Behavioural Segmentation in Pakistan

Using explainable machine learning (Random Forest / XGBoost + SHAP) and unsupervised segmentation (k-means), this project identifies who is excluded from Pakistan's digital financial system — and, more importantly, where formal access doesn't translate into actual usage.

## Headline finding

**44.3% of the sampled population has near-universal mobile phone ownership (98%) and majority internet use (62.3%), yet two-thirds of this group remain formally excluded from the financial system.** Access is clearly not the binding constraint for this segment — something else (trust, product design, documentation barriers) is. This pattern was found independently by unsupervised clustering, without being told about any predefined adoption categories, which strengthens it as a genuine structural finding rather than an artifact of how the target was built.

A second, related finding: gender is structurally intertwined with the access divide itself, not just a modest independent predictor — the most digitally-excluded cluster is 96.3% female, and gender remains a top-4 SHAP driver of adoption tier even after controlling for age, income, and workforce status.

## Data

- **Source:** [Global Findex Database 2025, Pakistan microdata](https://microdata.worldbank.org/index.php/catalog/7961) (Ref: `PAK_2024_FINDEX_v02_M`), World Bank Microdata Library.
- **Coverage:** 1,000 individual-level respondents, fieldwork May–Dec 2024, 183 raw variables.
- **Not included:** raw microdata file is not committed to this repo (large, and redistribution terms should be checked against the World Bank's microdata license). Download it yourself from the link above and place it in the project root before running the pipeline.

## Method

1. **Feature selection** — 20 variables across four construct groups (demographics, access, behaviour, resilience), selected from the Findex codebook. See `hypothesis_variable_selection.md` for the full mapping and rationale.
2. **Target construction** — a 4-level ordinal **Adoption Tier** (Excluded → Basic → Digitally Engaged → Advanced), built from account ownership and digital usage flags.
3. **Supervised modelling** — Logistic Regression → Decision Tree → Random Forest → XGBoost, evaluated on accuracy and F1 (macro + weighted), explained with SHAP.
   - **Note on leakage:** the variables used to *construct* the target (`account`, `dig_account`, `anydigpayment`, `saved`, `fin22a`, and their near-equivalents `account_fin`/`account_mob`) are explicitly excluded from the feature set. Leaving them in initially produced a spurious ~100% accuracy — a caught-and-fixed mistake worth noting for anyone reviewing this pipeline.
4. **Unsupervised segmentation** — k-means (k=2–5) on the same leakage-safe feature set, k selected via silhouette score (Davies-Bouldin and Calinski-Harabasz reported alongside), clusters profiled and visualized via PCA.
5. **Statistical validation** — chi-square tests of independence (cluster vs. gender, cluster vs. urban/rural) and Wilson score confidence intervals on headline proportions.
6. **Literature comparison** — results are compared directly against three anchor papers: Chowdhury et al. (2026, Bangladesh), Emuron et al. (2025, South Africa), and Saidy & Hassan (2025, 36 Sub-Saharan African countries).

## Results at a glance

| Model | Accuracy | F1 (macro) |
|---|---|---|
| XGBoost | 0.800 | 0.481 |
| **Random Forest (best)** | 0.708 | **0.486** |
| Decision Tree | 0.488 | 0.424 |
| Logistic Regression | 0.420 | 0.392 |

| Cluster | Size | Profile | Mean Adoption Tier |
|---|---|---|---|
| 0 — Digitally Excluded Women | 43.1% | 96.3% female, 51% rural, 9.7% internet use | 0.21 |
| 1 — Connected but Unengaged | 44.3% | 16.3% female, 98% mobile phone, 62.3% internet use | 0.80 |
| 2 — Advanced Adopters | 12.6% | Highest income/education, 84–98% on all behaviour flags | 2.92 |

## Repository structure

```
.
├── Findex_Microdata_2025_Pakistan.csv       # (not committed - download separately)
├── hypothesis_variable_selection.md          # Week 1: methods & variable mapping
├── 01_load_and_construct_target.py           # data loading, cleaning, target construction
├── 02_model_comparison.py                    # supervised modelling + SHAP
├── 03_segmentation.py                        # k-means segmentation + PCA
├── 04_statistical_tests.py                   # chi-square tests + confidence intervals
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
```

Each script reads the output of the previous one and saves its own CSV/PNG outputs to the project root.

## Hypotheses tested

- **H1 (access-usage gap):** Confirmed — both via a direct 6.6% dormant-account rate among account holders, and more strongly via the independently-derived Cluster 1 (44.3% of sample, high access, low usage).
- **H2 (rural vs. urban):** Urban adoption exceeds rural adoption, consistent with mainstream inclusion literature rather than Chowdhury et al.'s reversed Bangladesh finding.
- **H3 (drivers):** Behavioural variables (formal/informal saving) and gender outrank pure access variables (mobile phone, internet use) as SHAP-ranked predictors — a divergence from Saidy & Hassan's Sub-Saharan Africa findings, where access dominated.

## Limitations

- Sample size (n=1,000) limits reliability for small subgroups — the Basic tier (n=21) is essentially unpredictable by any model tested, a sample-size artifact rather than a modelling failure.
- `fin17b` (saved via mobile money) is partly confounded with mobile money account ownership itself due to Findex's survey routing logic; its high SHAP ranking should be read with this caveat.
- The k=3 cluster solution's silhouette score is only marginally better than k=2 — the 3-cluster structure is interpretable but not overwhelmingly dominant statistically.
- All language in the results is associational, not causal; Findex is a cross-sectional survey.

## Data citation

World Bank. 2025. *Global Findex Database 2025, Pakistan.* Ref: PAK_2024_FINDEX_v02_M. World Bank Microdata Library. https://microdata.worldbank.org/index.php/catalog/7961
