# 🚦 Smart Traffic Congestion Control System Optimization

*A Comprehensive Machine Learning Solution for Urban Traffic Management*

---

## 🧠 1. Project Title
**Smart Traffic Congestion Control System Optimization:**  
A Comprehensive Machine Learning Solution for Urban Traffic Management.

---

## 🛑 2. Problem Statement

### 🚧 The Challenge
Urban traffic congestion leads to:
- 💸 Economic losses (fuel, time, delivery delays)
- 🌫 Environmental pollution (vehicle emissions)
- 😩 Quality of life degradation (stress, delays)

### 🎯 Insights Sought
- Temporal congestion patterns (hourly, daily, weekly)
- Key contributors to congestion (weather, incidents, road type)
- Predictive models for congestion levels
- Real-time optimization strategies

### 🎯 Project Goals
- Predict congestion levels accurately
- Recommend optimization strategies
- Provide real-time, interpretable visualizations

---

## 📊 3. Dataset Overview

### 🗂️ Data Sources
- Synthetic Data via `generate_realistic_traffic_data()`
- Kaggle Integration via `xtraffic` adapter

### 📈 Dataset Summary
- Records: ~14,420  
- Features: 14  
- Roads: 20  
- Duration: 30 days (hourly)

### 📋 Feature Types
- **Numerical:** hour, volume, speed, precipitation, congestion, etc.
- **Categorical:** road_type  
- **Binary:** is_weekend, is_rush_hour, incident  
- **Datetime:** timestamp

### 🎯 Target Variable
- `congestion`: Float (0–1), normalized congestion score

---

## 🧹 4. Data Preprocessing

- ✅ **Missing Data:** Imputed using mean/median/mode  
- ⚠️ **Outlier Handling:** IQR-based removal  
- 📏 **Feature Scaling:** `StandardScaler`  
- 🔡 **Categorical Encoding:** One-Hot and Binary  
- 🔀 **Train-Test Split:** 80-20 stratified by road type  
- 🛠️ **Feature Engineering:** hour, day_of_week, is_weekend, etc.

---

## 🧪 5. Exploratory Data Analysis (EDA)

### 📅 Temporal Patterns
- Peak hours: 7–9 AM, 4–6 PM  
- Friday: Highest congestion  
- Sunday: Lowest congestion

### 🔥 Correlation Highlights
- `volume` ↔ `congestion`: +0.73  
- `speed` ↔ `congestion`: −0.82

### 📊 Feature Insights
- **Highways:** High volume, moderate congestion  
- **Incidents:** Raise congestion by ~150%

---

## ⚙️ 6. Model Selection and Justification

| Model         | Type         | Purpose                       | Justification                            |
|---------------|--------------|-------------------------------|------------------------------------------|
| k-NN          | Classification | Categorize congestion levels | Pattern-based similarity                 |
| Random Forest | Regression   | Predict congestion score      | Handles non-linearity & noise well       |
| XGBoost       | Regression   | Precise prediction            | High performance on tabular data         |
| Knowledge Graph | Rule-based | Recommend strategies          | Domain knowledge encoding                |

---

## 📈 7. Model Training

- 🔁 **Cross-Validation:** 5-Fold CV  
- ⚙️ **Hyperparameters:**
  - Random Forest: `n_estimators=100`, `max_depth=10`
  - XGBoost: `learning_rate=0.1`, `max_depth=6`, `n_estimators=150`
  - k-NN: `k=5`
- 🕒 **Resources:** CPU-only, <1 min training

---

## 🧮 8. Evaluation Metrics

### 📏 Regression Metrics

| Model         | RMSE    | MAE     | R²     |
|---------------|---------|---------|--------|
| k-NN          | 0.0867  | 0.0652  | 0.842  |
| Random Forest | 0.0613  | 0.0418  | 0.927  |
| XGBoost       | 0.0542  | 0.0394  | 0.952  |

### 🔍 Classification Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| k-NN  | 0.872    | 0.864     | 0.867  | 0.865    |

### 📊 Visualizations
- ROC Curves
- Precision-Recall Curves
- Residual Plots

---

## 🛠️ 9. Hyperparameter Tuning
- Tool: `GridSearchCV`
- Tuned: `n_estimators`, `max_depth`, `learning_rate`
- **Outcome:** RMSE reduced by ~12%

---

## 🧩 10. Final Model Pipeline

```python
pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('regressor', XGBRegressor(...))
])
pipeline.fit(X_train, y_train)
