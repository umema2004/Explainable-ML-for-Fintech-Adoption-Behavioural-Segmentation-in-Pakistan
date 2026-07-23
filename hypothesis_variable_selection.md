# Hypothesis & Variable Selection
## Explainable ML for Fintech Adoption & Behavioural Segmentation in Pakistan

**Dataset:** Global Findex Database 2025, Pakistan (Ref: PAK_2024_FINDEX_v02_M) — 1,000 respondents, 183 variables, collected May–Dec 2024.

---

## 1. Research Question

Which socio-economic and access factors predict digital financial adoption in Pakistan, and do distinct behavioural segments exist among the population — particularly a gap between formal access and actual digital usage?

## 2. Hypotheses

**H1 — Access–usage gap:** Does having a formal account predict actual digital engagement, or is there a substantial "dormant account" segment (the structural analogue of Chowdhury et al.'s knowledge–behaviour gap)?
- *Testable via:* cross-tab of `account` vs. `dig_account`/`anydigpayment`; the "account=Yes, dig_account=No" cell size is the headline number.

**H2 — Rural vs. urban:** Does rural adoption exceed urban adoption once mobile access is controlled for (Chowdhury et al.'s finding), or does the more common urban advantage hold?
- *Testable via:* `urbanicity` as a feature + SHAP dependence; cluster profiling by `urbanicity`.

**H3 — Drivers comparison:** Do account ownership and mobile/internet access dominate as predictors (Saidy & Hassan, Sub-Saharan Africa), or does something else (income, gender, age, education) dominate in Pakistan specifically?
- *Testable via:* SHAP feature importance ranking from the best-performing model.

## 3. Selected Variables (Findex Pakistan 2024 codebook)

### Demographics
| Variable | Description |
|---|---|
| `female` | Respondent is female |
| `age` | Respondent age |
| `educ` | Education level |
| `inc_q` | Income quintile (within-economy) |
| `emp_in` | In workforce |
| `urbanicity` | Lives in rural area |

### Access
| Variable | Description |
|---|---|
| `account` | Has an account (any) |
| `account_fin` | Has account at financial institution |
| `account_mob` | Has mobile money account |
| `dig_account` | Has a digitally enabled account |
| `con1` | Has a mobile phone |
| `internet_use` | Used internet in past 3 months |

### Behaviour
| Variable | Description |
|---|---|
| `anydigpayment` | Made/received a digital payment |
| `merchantpay_dig` | Made digital merchant payment |
| `saved` | Saved (any method) |
| `fin17b` | Saved using mobile money |
| `fin17c` | Saved informally (club/person) |
| `borrowed` | Borrowed (any source) |
| `fin22a` | Borrowed from bank/formal institution |
| `fin22a_1` | Borrowed from mobile money provider |

### Resilience
| Variable | Description |
|---|---|
| `fin24a` | Difficulty raising emergency funds within 30 days |
| `fin24b` | How long household could cover expenses if income lost |

**Note:** 20 variables selected, slightly above the 10–15 target. Final trim will happen in Week 3 via correlation check / feature importance, keeping the set that best balances signal and interpretability on ~1,000 rows.

## 4. Target Construction — Adoption Tier

A four-level ordinal target built from raw variables (not directly present in the data):

| Tier | Label | Condition |
|---|---|---|
| 0 | Excluded | `account` = No |
| 1 | Basic | `account` = Yes, and `dig_account` = No and `anydigpayment` = No |
| 2 | Digitally Engaged | `dig_account` = Yes or `anydigpayment` = Yes |
| 3 | Advanced | Tier 2 conditions + (`saved` = Yes or `borrowed` = Yes) + 2+ product types (e.g., account + card + digital payment) |

Used two ways, mirroring the Chowdhury et al. dual approach:
- **Supervised:** predict Adoption Tier with Random Forest / XGBoost, explain with SHAP.
- **Unsupervised:** k-means on the same features, independent of tier labels — check whether clusters resemble the defined tiers, or reveal a distinct "account-holder who never uses it" cluster (the H1 access–usage gap group).

## 5. Positioning Statement

Chowdhury et al. (2026) demonstrated explainable ML and behavioural segmentation for financial literacy in Bangladesh; Emuron et al. (2025) and Saidy & Hassan (2025) modelled fintech adoption in Africa using simpler ML toolkits without segmentation. This project combines segmentation methodology with adoption prediction for Pakistan — a combination not yet applied to South Asian data.
