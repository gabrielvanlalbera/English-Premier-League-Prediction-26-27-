# EPL 2026/27 Match Outcome Prediction & League Forecast

A data analyst / machine learning project for predicting the **remaining 2026/27 English Premier League fixtures** and estimating the most likely league champion from historical match data.

## Project goal

This project answers two questions:

1. **Who is most likely to win each remaining 2026/27 match?**
2. **Which club is most likely to win the 2026/27 league?**

The main classification target is `FullTimeResult`:

- `H` = Home win
- `D` = Draw
- `A` = Away win

Linear Regression is used as an interpretable baseline for goal-difference prediction. XGBoost is the main H/D/A classifier and produces probability estimates for all three outcomes.

## Data integration

Three user-supplied datasets are combined:

- `epl_final.csv`
- `epl-allseasons-matchstats.csv`
- `Book1.xlsx`

The pipeline standardizes team names and deduplicates matches using a date + home team + away team key.

The merged historical dataset contains **9,579 completed matches**. The integration adds **199 completed records** that were not present in the original file, including the previously incomplete 2024/25 games and the available completed 2025/26 games.

No invented results are added. The 2003/04 and 2004/05 seasons remain shorter because the supplied auxiliary data does not contain the missing games for those seasons.

## Features

### Pre-match model

The full-season forecast uses information that can be known before kickoff:

- Home and away team
- Elo ratings
- Last-5 points per game
- Last-5 win rate
- Recent goals scored and conceded
- Recent goal difference
- Home/away form
- Rest-day difference
- Head-to-head history
- Calendar features

### In-match model

A separate model follows the original project idea and uses statistics available during a match:

- Half-time score
- Shots
- Shots on target
- Corners
- Fouls
- Yellow cards
- Red cards
- xG

The **final score is never used as an input** because it is the ground truth for the result.

## Referee analysis

Referee history is analyzed using:

- Average fouls per match
- Average yellow cards
- Average red cards
- Home-vs-away yellow-card bias
- Historical H/D/A rates

These are treated as historical associations rather than causal effects. Referee appointments are not required for the full-season pre-match forecast because future assignments are not known for every fixture in the current project snapshot.

## 2026/27 forecast

The released 2026/27 EPL calendar contains **380 fixtures** across 20 teams, running from August 2026 to May 2027. In this project snapshot, Matchweeks 1–5 are completed and the remaining **330 fixtures from MW6–MW38** are forecast. Fixtures can be rescheduled, so the schedule should be refreshed before publishing a new forecast.

The complete prediction file is:

`results/2026_27_all_upcoming_match_predictions.csv`

A complete season table combining completed and upcoming fixtures is:

`data/processed/2026_27_full_season_fixtures_predictions.csv`

## League winner forecast

The title race is simulated with **20,000 Monte Carlo season simulations** using the XGBoost 1X2 probabilities for the remaining fixtures.

Current model result as of **8 October 2026**:

| Team | Title probability | Expected final points | Top-4 probability |
|---|---:|---:|---:|
| **Manchester City** | **80.54%** | **82.2** | **99.56%** |
| Arsenal | 14.86% | 71.3 | 90.64% |
| Liverpool | 2.81% | 63.9 | 61.15% |

The title probabilities are model estimates, not guarantees. The current simulation holds fixture probabilities fixed at the 8 October model state rather than dynamically updating team strength after every simulated result.

Outputs:

- `results/2026_27_title_race_monte_carlo.csv`
- `results/2026_27_expected_final_table.csv`
- `results/model_winner_summary.json`

## Backtesting

A chronological 2024/25 hold-out is used instead of a random split.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Linear Regression — pre-match | 41.05% | 0.367 |
| XGBoost — pre-match | **46.58%** | 0.346 |
| Linear Regression — in-match | **65.43%** | **0.598** |
| XGBoost — in-match | 62.86% | 0.571 |

The in-match models are trained only on rows where the required live statistics are available. Older seasons without those fields are not artificially filled.

## Repository structure

```text
epl-2026-27-full-project/
├── data/
│   ├── raw/
│   │   ├── epl_final.csv
│   │   ├── epl-allseasons-matchstats.csv
│   │   └── Book1.xlsx
│   └── processed/
│       ├── epl_combined_historical.csv
│       ├── engineered_historical_features.csv
│       ├── referee_impact_profile.csv
│       └── 2026_27_full_season_fixtures_predictions.csv
│
├── notebooks/
│   └── 01_EPL_2026_27_Full_Forecast.ipynb
│
├── models/
│   ├── prematch_xgboost.joblib
│   ├── prematch_linear_regression.joblib
│   ├── inmatch_xgboost.joblib
│   └── inmatch_linear_regression.joblib
│
├── results/
│   ├── 2026_27_all_upcoming_match_predictions.csv
│   ├── 2026_27_expected_final_table.csv
│   ├── 2026_27_title_race_monte_carlo.csv
│   ├── referee_profiles_20plus_matches.csv
│   └── model_metrics_*.csv
│
├── src/
│   ├── epl_pipeline.py
│   └── rebuild_forecast.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open:

`notebooks/01_EPL_2026_27_Full_Forecast.ipynb`

To rebuild the processed tables, models and forecast from the repository data, run:

```bash
python src/rebuild_forecast.py
```

## Current-season update workflow

After each completed 2026/27 matchweek:

1. Add the completed results.
2. Rebuild Elo and rolling historical features.
3. Retrain/refit the models.
4. Remove completed fixtures from the future set.
5. Regenerate all remaining 1X2 predictions.
6. Re-run the Monte Carlo title simulation.

## Schedule references

2026/27 fixture list (20 teams, 380 matches):
https://raw.githubusercontent.com/openfootball/england/master/2026-27/1-premierleague.txt

Fixture/results export page:
https://fixturedownload.com/results/epl-2026

## Disclaimer

This is a football analytics and machine learning project. Model probabilities are statistical estimates and should not be treated as guaranteed outcomes or betting advice.
