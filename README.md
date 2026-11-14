# 🚀 Spaceship Titanic: Predicting Transported Passengers

A machine learning solution to the Kaggle **[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)** competition. The task is to predict whether each passenger was **transported to an alternate dimension** after the spaceship collided with a spacetime anomaly.

The project covers **logic-based missing-value imputation, feature engineering, and gradient-boosting models** (XGBoost, LightGBM and CatBoost). The final model is a **CatBoost classifier tuned with Optuna**.

---

## 🏆 Results

| | |
|---|---|
| **Final model** | CatBoost, tuned with Optuna (100 trials) |
| **5-fold CV accuracy** | **81.84% ± 0.67%** |
| **Kaggle submission accuracy** | **80.789%** |

**Baseline comparison.** All three models used the same preprocessing pipeline, 500 trees, depth 5 and learning rate 0.05, scored with 5-fold stratified CV.

| Model | CV accuracy |
|---|---|
| LightGBM | 81.04% ± 0.46% |
| XGBoost | 81.19% ± 0.79% |
| CatBoost | 81.30% ± 0.50% |
| **CatBoost + Optuna** | **81.84% ± 0.67%** |

---

## 📌 The data

| | |
|---|---|
| **Training passengers** | 8,693 |
| **Test passengers** | 4,277 |
| **Target** | `Transported` (True / False), balanced at 50.4% True |
| **Missing values** | About 2% in almost every column |

The raw features are:
- `PassengerId`, which encodes group and position in the group
- `HomePlanet`, `CryoSleep`, `Cabin` (deck/number/side), `Destination`, `Age`, `VIP`
- Five onboard spending columns: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- `Name`

---

## 🔍 Key findings from EDA

- **Cabin deck reveals home planet.** Decks A, B, C and T are over 98% Europa, and deck G is 99.7% Earth. This let missing `HomePlanet` values be filled by logic instead of guessed.
- **CryoSleep means no spending.** Passengers in cryosleep are confined to their cabins, so their spending must be zero. A missing `CryoSleep` value can be inferred from spending.
- **Spending is a strong signal.** Passengers who were *not* transported spent more than twice as much on average (≈2,004) as those who were (≈886).
- **Groups travel together.** Passengers with the same group ID usually share a home planet, deck and side, which makes them reliable sources for imputation.

---

## ⚙️ Approach

### 1. Logic-based imputation (`src/cleaned_data.py`, `overview.ipynb`)
Missing values were filled using the structure of the data rather than global averages:
- **HomePlanet:** filled with the group's most common value, then by deck (A/B/C/T → Europa, G → Earth), then by destination
- **CryoSleep:** inferred from spending (zero spend → True)
- **Age:** filled with the median for the passenger's HomePlanet and CryoSleep combination
- **Deck and Side:** filled with the group's most common value, then the home planet's
- **Destination:** filled with the most common destination for the passenger's home planet
- **VIP:** inferred from total spend
- **Spending:** missing values set to 0, and forced to 0 for passengers in cryosleep

### 2. Feature engineering (`Spaceship_titanic_analysis_pipeline.ipynb`)
- **From IDs and cabins:**
  - `Group` and `PersonInGroup` are split out of `PassengerId`.
  - `Deck`, `CabinNum` and `Side` are split out of `Cabin`.
- **Spending:**
  - Totals: `TotalSpend`, `LuxurySpend` (Spa + VRDeck + RoomService) and `BasicSpend` (FoodCourt + ShoppingMall)
  - Derived: a `NoSpend` flag and `SpendPerAge`
  - All spending features are log-transformed to reduce skew.
- **Group features:**
  - `GroupSize`, `IsAlone` and `FamilySize` (passengers who share a last name)
  - These are computed **from the training set only** and then mapped onto the test set, so no test information leaks in.
- **Other:**
  - `AgeBin`
  - `Cryo_Child`, a flag for children in cryosleep

In total there are **24 features**: 16 numeric and 4 categorical, plus 4 integer flags. They go through a scikit-learn `ColumnTransformer` that applies median or mode imputation and one-hot encoding.

### 3. Modelling and tuning
- **Validation:** 5-fold stratified cross-validation, with every model wrapped in a scikit-learn `Pipeline` so preprocessing is refit inside each fold.
- **Optuna:** a 100-trial TPE search over 10 CatBoost hyperparameters, maximising mean CV accuracy. The best settings were 523 iterations, depth 5 and learning rate ≈ 0.052.
- **Final model:** retrained on all 8,693 passengers to produce the submission.

**Top features** (CatBoost importance):
1. `LuxurySpend_log` (24.7)
2. `Deck` (13.6)
3. `CabinNum` (11.6)
4. `HomePlanet` (8.4)
5. `BasicSpend_log` (6.0)

---

## 📁 Repository structure

```
Spaceship_titanic/
├── overview.ipynb                              # First exploration + step-by-step imputation logic
├── EDA and Feature Engineering.ipynb           # Visual EDA and early feature ideas
├── Data Modeling.ipynb                         # Earlier XGBoost / CatBoost experiments
├── Spaceship_titanic_analysis_pipeline.ipynb   # ⭐ Final pipeline: FE → baselines → Optuna → submission
├── src/
│   └── cleaned_data.py                         # Reusable CleanedData imputation class
└── catboost_info/                              # CatBoost training logs (auto-generated)
```

## 🚀 How to run

1. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and put them in a `data/` folder.
2. Install the dependencies:
   ```bash
   pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn
   ```
3. In the first cell of each notebook, replace the absolute `C:\Users\...` paths with `data/train.csv` and `data/test.csv`.
4. Run `Spaceship_titanic_analysis_pipeline.ipynb`. It writes `data/submission.csv`.

> ⏱️ The Optuna search (100 trials × 5-fold CV) took about 1.5–2 hours on a laptop. Lower `n_trials` for a quicker run.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · Matplotlib · Seaborn

## 🔮 Next steps

- [ ] Ensemble CatBoost, LightGBM and XGBoost (by averaging probabilities or stacking)
- [ ] Try the neural network (MLP) that the pipeline notebook sets up but doesn't train yet
- [ ] Use group-level features from the combined train and test data, such as shared cabins
- [ ] Remove the absolute file paths and the `catboost_info/` logs from the repo

---

## 👤 Author

**Divyesh Bhavsar**
GitHub: [@DivyeshRBhavsar](https://github.com/DivyeshRBhavsar)
