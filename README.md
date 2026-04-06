# Spotify Churn Propensity Modeling

Logistic-regression models that identify users at risk of churning, predict which free users are likely to subscribe, and flag premium users at risk of canceling — built on the [Spotify User Behavior and Pattern](https://www.kaggle.com/datasets/sahilislam007/spotify-user-behavior-and-pattern) dataset from Kaggle (50,000 users).

## Business Questions

| # | Question | Proxy Target Used |
|---|----------|-------------------|
| 1 | **Which users are likely to become inactive?** | `inactive_3_months_flag` — 1 if a user had 3+ consecutive months of inactivity |
| 2 | **Which free users are likely to subscribe?** | `ad_conversion_to_subscription` — 1 if a Free-tier user converted after an ad interaction |
| 3 | **Which premium users are at risk of canceling?** | `subscription_status == "Inactive"` among paid-plan users |

> **Note:** This dataset does not contain true churn labels with timestamps. The targets above are transparent proxy labels derived from available behavioral flags. See the notebook for full documentation of the logic and caveats.

## Project Structure

```
├── notebook.ipynb       # Full analysis — EDA, modeling, evaluation, business recommendations
└── README.md
```

Kaggle credentials are required for the dataset download — see [kagglehub docs](https://github.com/Kaggle/kagglehub#authenticate).

## Methodology

- **Feature engineering:** account tenure, premium flag, listen-to-skip engagement ratio, plus all original columns (age, country, genre, device, etc.)
- **Preprocessing:** `StandardScaler` for numerics, `OneHotEncoder` for categoricals, built as reusable `sklearn` pipelines
- **Modeling:** Logistic Regression with `class_weight="balanced"`, 80/20 stratified split
- **Evaluation:** ROC-AUC, precision, recall, F1, confusion matrix, top coefficients by magnitude

## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Inactivity Risk | 0.5022 | 0.2197 | 0.4876 | 0.3029 |
| Free to Subscribe | **0.8520** | 0.2402 | 1.0000 | 0.3873 |
| Premium Churn Risk | 0.4990 | 0.1651 | 0.5080 | 0.2492 |

- **Free to Subscribe (AUC = 0.85)** is the strongest model — `ad_interaction` is the dominant predictor. Directly actionable for targeting conversion campaigns.
- **Inactivity Risk** and **Premium Churn Risk** perform at random chance (AUC ~ 0.50), indicating the dataset's status labels are independent of behavioral features — consistent with synthetic data generation. A production model would require longitudinal engagement signals.

## Business Recommendations

1. **Free-to-paid conversion:** Use propensity scores to prioritize premium trial offers and targeted ad campaigns toward high-score Free users.
2. **Inactivity & premium churn:** The available features cannot reliably predict these outcomes. Production models would need session-level engagement logs, listening trend declines, support tickets, and payment failure events.
3. **Ad strategy:** Coefficient analysis shows ad interaction is the dominant conversion signal — improving ad targeting is the highest-leverage investment.

## Caveats

- Proxy targets, not true churn labels — models learn correlates of the proxy, not causal drivers
- Cross-sectional snapshot — no temporal dynamics; production models need time-series features
- Synthetic dataset — two near-random models suggest labels were assigned independently of behavior
- Logistic regression chosen for interpretability; ensemble methods could improve AUC but won't fix data limitations
