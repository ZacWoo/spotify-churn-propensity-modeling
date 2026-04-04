# Spotify Churn Propensity Modeling

Logistic-regression models that identify users at risk of churning, predict which free users are likely to subscribe, and flag premium users at risk of canceling — built on the [Spotify User Behavior and Pattern](https://www.kaggle.com/datasets/sahilislam007/spotify-user-behavior-and-pattern) dataset from Kaggle (50,000 users).

## Business Questions

| # | Question | Proxy Target Used |
|---|----------|-------------------|
| 1 | **Which users are likely to become inactive?** | `inactive_3_months_flag` — 1 if a user had 3+ consecutive months of inactivity |
| 2 | **Which free users are likely to subscribe?** | `ad_conversion_to_subscription` — 1 if a Free-tier user converted after an ad interaction |
| 3 | **Which premium users are at risk of canceling?** | `subscription_status == "Inactive"` among paid-plan users |

> **Important caveat:** This dataset does not contain true churn event labels with timestamps. The targets above are transparent *proxy* labels derived from available behavioral flags. Model outputs should be interpreted as propensity scores indicating relative risk, not ground-truth churn probabilities. See [Caveats](#caveats) for details.

## Project Structure

```
spotify-churn-propensity-modeling/
├── main.py                    # Single entry point — run the full pipeline
├── src/
│   ├── data_loader.py         # Download dataset via Kaggle API
│   ├── preprocessing.py       # Feature engineering & sklearn pipelines
│   ├── models.py              # Train & evaluate logistic regression
│   └── visualization.py       # ROC curves, confusion matrices, coefficients
├── outputs/
│   ├── metrics/               # JSON files with per-model metrics
│   └── plots/                 # PNG plots (ROC, confusion matrix, coefficients)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/spotify-churn-propensity-modeling.git
cd spotify-churn-propensity-modeling

# 2. Create a virtual environment (optional but recommended)
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure Kaggle credentials are configured
#    (see https://github.com/Kaggle/kagglehub#authenticate)

# 5. Run the full pipeline
python main.py
```

The script will:
1. Download the dataset via `kagglehub` (cached after first run).
2. Engineer features (account age, listen-to-skip ratio, premium flag).
3. Train three logistic-regression models with balanced class weights.
4. Print evaluation metrics and save them to `outputs/metrics/`.
5. Save ROC curves, confusion matrices, and coefficient charts to `outputs/plots/`.

## Methodology

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `account_age_days` | Days between signup date and dataset snapshot |
| `is_premium` | Binary flag for paid subscription tiers |
| `listen_to_skip_ratio` | `avg_listening_hours_per_week / (avg_skips_per_day + 1)` |

Original dataset columns (age, country, genre, device, playlists created, music suggestion rating, etc.) are also included as features.

### Preprocessing Pipeline

- **Numeric features** are standardized with `StandardScaler`.
- **Categorical features** are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`.
- Pipelines are built with `sklearn.pipeline.Pipeline` for reproducibility.

### Modeling

- **Algorithm:** Logistic Regression (`solver="lbfgs"`, `max_iter=1000`).
- **Class balancing:** `class_weight="balanced"` to handle imbalanced targets.
- **Train/test split:** 80/20, stratified by target.

### Evaluation Metrics

Each model reports:
- **ROC-AUC** — discriminative ability across all thresholds
- **Precision** — of predicted positives, how many are correct
- **Recall** — of actual positives, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — true/false positive/negative counts
- **Top Coefficients** — most influential features by magnitude

## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Inactivity Risk | 0.5022 | 0.2197 | 0.4876 | 0.3029 |
| Free to Subscribe | **0.8520** | 0.2402 | 1.0000 | 0.3873 |
| Premium Churn Risk | 0.4990 | 0.1651 | 0.5080 | 0.2492 |

### Interpretation

- **Free to Subscribe (AUC = 0.85)** is the strongest model. The top predictor is `ad_interaction`, which logically gates the conversion target — users who interacted with ads are far more likely to convert. This model is actionable for targeting conversion campaigns.
- **Inactivity Risk (AUC ~ 0.50)** and **Premium Churn Risk (AUC ~ 0.50)** perform near random chance. This is an honest and important finding: the behavioral features in this dataset have negligible linear correlation with the inactivity flag or subscription status. This strongly suggests the dataset's inactive/status labels were assigned independently of the behavioral columns (consistent with synthetic data generation). In a real-world setting, churn prediction requires longitudinal engagement signals that this cross-sectional snapshot does not provide.

After running `python main.py`, check `outputs/metrics/` for full JSON metrics and `outputs/plots/` for ROC curves, confusion matrices, and coefficient charts.

## Business Recommendations

1. **Free-to-paid conversion (actionable now):** The conversion model identifies Free users with high subscribe propensity. Target these users with premium trial offers, reduced-friction upgrade flows, and personalized ad campaigns.
2. **Inactivity & premium churn (requires better data):** The near-random AUC on these models means the available features cannot reliably separate churners from non-churners. To build production-grade churn models, invest in collecting:
   - Session-level engagement logs (daily/weekly active usage trends)
   - Listening streak and dropout patterns over time
   - Support ticket / complaint history
   - Payment failure events
3. **Feature investment:** Coefficient analysis on the conversion model shows ad interaction is the dominant signal. Improving ad targeting and ad-to-trial funnels is the highest-leverage product investment the data supports.

## Caveats

- **Proxy targets, not true labels.** The dataset lacks timestamped churn events. The three targets are constructed from behavioral flags and current status columns. This means the models learn *correlates of the proxy*, not causal drivers of churn.
- **Synthetic dataset limitations.** Two of three models produce near-random AUC, indicating that the inactivity and status labels are likely independent of behavioral features in this dataset. This is consistent with synthetic data generation and does not reflect what a real churn model would achieve with production Spotify data.
- **Cross-sectional data.** All features represent a single snapshot in time. A production churn model would benefit from longitudinal behavioral sequences (e.g., week-over-week listening decline).
- **Logistic regression only.** This project uses logistic regression for interpretability. Ensemble methods (gradient boosting, random forests) could improve performance but would not overcome the fundamental data limitations noted above.
- **No external validation.** There is no held-out temporal validation set. Real-world deployment would require monitoring model performance on future data.

## License

MIT
