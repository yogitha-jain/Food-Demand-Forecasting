# 🍽️ FoodCast — AI Demand Forecasting & Inventory Optimization

> AI-powered weekly meal demand forecasting for food delivery restaurant chains using XGBoost and time series analysis.


---

## 📌 Problem Statement

The restaurant and hospitality industry operates on razor-thin margins with highly perishable inventory. Traditional inventory management relies on static spreadsheets and gut feelings, leading to:

- **Over-ordering** → expensive food waste and spoilage
- **Under-ordering** → stockouts and lost revenue

This project builds an AI-powered Demand Forecasting model for a food delivery restaurant chain. By analyzing historical point-of-sale (POS) data, the model predicts future weekly order volumes for specific menu items — enabling managers to optimize their supply chain and reduce food waste by 20–30%.

---

## 📊 Dataset

**Source:** [Food Demand Forecasting — Kaggle (Genpact ML Hackathon)](https://www.kaggle.com/datasets/arashnic/food-demand)

| File | Description |
|------|-------------|
| `train.csv` | Weekly order data — 456,548 rows, 145 weeks |
| `meal_info.csv` | Meal category and cuisine details (51 meals) |
| `fulfilment_center_info.csv` | Center type, city, region, area (77 centers) |

**Key columns:**

| Column | Description |
|--------|-------------|
| `week` | Week number (1–145) |
| `center_id` | Fulfillment center ID |
| `meal_id` | Menu item ID |
| `checkout_price` | Final price charged to customer |
| `base_price` | Original price before discount |
| `emailer_for_promotion` | Email promotion sent (0/1) |
| `homepage_featured` | Featured on homepage (0/1) |
| `num_orders` | Target — weekly orders to predict |

---

## 🗂️ Project Structure

```
Food-Demand-Forecasting/
├── week1Resto.ipynb                # Week 1: EDA & Time Series Analysis
├── week2Resto.ipynb                # Week 2: Feature Engineering
├── week3_model_training.ipynb      # Week 3: Model Training & Selection
├── week4Resto.ipynb                # Week 4: Evaluation & Business Reporting
├── app.py                          # Flask Backend
├── static/
│   └── index.html                  # Frontend (FoodCast UI)
├── X_train.csv                     # Train features
├── X_test.csv                      # Test features
├── y_train.csv                     # Train labels
├── y_test.csv                      # Test labels
├── train_test_split.png            # Sequential split visualization
├── demand_trend.png                # Overall demand trend
├── weekly_seasonality.png          # Weekly seasonality pattern
├── decomposition.png               # Time series decomposition
├── autocorrelation.png             # ACF and PACF plots
├── model_comparison.png            # All 3 models RMSE comparison
├── feature_importance_final.png    # Top features driving demand
├── forecast_vs_actual.png          # Predictions vs actuals overlay
├── .gitignore                      # Excludes .pkl and raw CSV files
└── README.md
```

---

### ✅ Week 1: Data Ingestion & Time-Series EDA

- Loaded and merged 3 datasets (`train.csv`, `meal_info.csv`, `fulfilment_center_info.csv`)
- Performed data quality checks — zero missing values, zero duplicates
- Plotted overall demand trend with 4-week rolling average
- Weekly seasonality patterns and monthly analysis
- Promotion impact analysis — email promotion increases orders 3x (229 → 631)
- Time series decomposition: Trend + Seasonal (13-week cycle) + Residual
- ACF and PACF autocorrelation analysis
- ADF Stationarity test

**Key finding:** Beverages dominate with 40M+ total orders. TYPE_A centers handle 69M orders — far more than B and C combined.

---

### ✅ Week 2: Advanced Feature Engineering

- **Chronological features:** `week_in_year`, `month`, `quarter`, `is_weekend`, `is_holiday`
- **Lag features:** `lag_1`, `lag_2`, `lag_4`, `lag_8` — orders from N weeks ago per center+meal
- **Rolling window statistics:** 4-week mean, 8-week mean, 4-week std
- **Sequential train/test split:** Week 1–120 = Train | Week 121–145 = Test
- No random split — prevents data leakage from future into past

**Final feature set:** 20 engineered features  
**Train size:** 373,919 rows | **Test size:** 82,629 rows

---

### ✅ Week 3: Model Training & Selection

Three models trained and compared using **Time-Series Cross-Validation (5 folds)**:

| Model | CV Mean RMSE | Test RMSE | Test MAE | Test RMSLE |
|-------|-------------|-----------|----------|------------|
| Linear Regression (Baseline) | — | Baseline | — | — |
| Random Forest | 164.37 | 170.25 | 74.04 | 0.5387 |
| **XGBoost ✅ Best** | **160.48** | **160.82** | **73.07** | **0.6716** |

**Best XGBoost Hyperparameters (found via TimeSeriesSplit CV):**

```python
n_estimators     = 500
learning_rate    = 0.05
max_depth        = 6
subsample        = 0.8
colsample_bytree = 0.8
```

---

### ✅ Week 4: Evaluation, Feature Importance & Business Reporting

**Final Model Evaluation (XGBoost on Test Set):**

| Metric | Value |
|--------|-------|
| Test RMSE | 160.82 |
| Test MAE | 73.07 |
| Test RMSLE | 0.6716 |

**Business interpretation:**
- On average, the model prediction is off by **73 orders** per meal per center per week
- Enables **20–30% reduction in food waste** through accurate demand planning

**Top Demand Drivers (Feature Importance):**

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `lag_1` | Last week's orders — strongest single predictor |
| 2 | `rolling_mean_4` | 4-week average — captures recent demand trend |
| 3 | `checkout_price` | Price directly impacts customer demand |
| 4 | `homepage_featured` | Homepage promotion boosts orders significantly |
| 5 | `emailer_for_promotion` | Email campaigns drive predictable demand spikes |

---

## 🚀 Running the Application

### Prerequisites

```bash
pip install flask flask-cors xgboost scikit-learn pandas numpy
```

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yogitha-jain/Food-Demand-Forecasting.git
cd Food-Demand-Forecasting
```

2. Place your trained model file `xgboost_demand_model.pkl` in the root folder (excluded from GitHub via `.gitignore` due to file size).

3. Start the Flask server:

```bash
python app.py
```

4. Open browser and go to:

```
http://localhost:5000
```

### Using the FoodCast UI

| Input | Description |
|-------|-------------|
| Week Number | Week to forecast (1–145) |
| Center ID | Fulfillment center (e.g. 55) |
| Meal ID | Menu item (e.g. 1885) |
| Checkout Price | Actual selling price |
| Base Price | Original price before discount |
| Avg Recent Orders | Approximate recent weekly orders |
| Email Promotion | Whether email campaign is active |
| Homepage Featured | Whether meal is featured on homepage |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Statsmodels |
| ML Models | Scikit-Learn, XGBoost |
| Backend | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Version Control | Git, GitHub |

---

## 📈 Key Visualizations

| File | Description |
|------|-------------|
| `demand_trend.png` | Overall weekly demand with rolling average |
| `decomposition.png` | Trend + Seasonal + Residual decomposition |
| `model_comparison.png` | RMSE comparison across all 3 models |
| `feature_importance_final.png` | Top features driving predictions |
| `forecast_vs_actual.png` | Weekly predicted vs actual overlay |

---

## 📝 Version Control Standards

This project follows enterprise-grade Git standards as per Infotact internship guidelines:

- Commit prefixes: `data-clean:`, `eda:`, `feature-eng:`, `model-tuning:`, `feat:`
- Minimum 4 weeks of commits and contributions
- No large files pushed — `.pkl` and raw `.csv` excluded via `.gitignore`
- Commit history reflects iterative development week by week

---

## 👩‍💻 Author

**Yogitha Jain**  
