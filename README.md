# 🏠 House Price Prediction

A beginner-friendly end-to-end Machine Learning project that predicts house prices using the California Housing dataset.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

## 📋 Overview

| | |
|---|---|
| **Problem** | Can we predict the median house price in a California district based on features like income, location, and house age? |
| **Dataset** | California Housing Dataset (20,640 samples, 8 features) — built into Scikit-Learn |
| **Approach** | Linear Regression → Decision Tree → Random Forest (comparing 3 models) |
| **Best Model** | Random Forest with R² score of ~0.81 |

## 📊 Key Visualizations

### Feature Correlation Heatmap
Shows which features are most related to house prices. **Median income** has the strongest correlation!

### Actual vs Predicted Prices
Compares what the model predicted vs the real prices — the closer to the diagonal line, the better.

### Feature Importance
Shows which features the model relies on most to make predictions.

## 🛠️ Tech Stack
- **Python 3.8+** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Visualizations
- **Scikit-Learn** — Machine Learning models

## 🚀 How to Run

**Step 1:** Clone this repository
```bash
git clone https://github.com/nivasbheemineni123-cmd/house-price-prediction.git
cd house-price-prediction
```

**Step 2:** Install the required packages
```bash
pip install -r requirements.txt
```

**Step 3:** Run the project
```bash
python house_price_prediction.py
```

This will train the models, print results, and save visualizations in the `plots/` folder.

## 📁 Project Structure
```
house-price-prediction/
├── README.md                    ← You are here
├── requirements.txt             ← Required Python packages
├── house_price_prediction.py    ← Main Python script
└── plots/                       ← Generated visualizations
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

## 📝 What I Learned
- How to load and explore a dataset using Pandas
- How to visualize data to find patterns
- How to train and compare multiple ML models
- How to evaluate models using metrics like R² and RMSE
- How to identify which features matter most for predictions

## 📬 Contact
**Nivas Bheemineni** — [GitHub](https://github.com/nivasbheemineni123-cmd)
