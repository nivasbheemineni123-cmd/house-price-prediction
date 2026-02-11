"""
🏠 House Price Prediction
=========================
This script predicts house prices using the California Housing dataset.
We'll go through the full Data Science workflow:
  1. Load the data
  2. Explore & visualize it (EDA)
  3. Prepare it for modeling
  4. Train 3 different models
  5. Compare and evaluate them

Author: Nivas Bheemineni
"""

# ============================================================
# STEP 0: Import Libraries
# ============================================================
# These are the tools we need. Think of them like apps on your phone.

import pandas as pd                  # For working with tables of data
import numpy as np                   # For math operations
import matplotlib.pyplot as plt      # For creating charts/graphs
import seaborn as sns                # For prettier charts
import os                            # For creating folders

from sklearn.datasets import fetch_california_housing   # Our dataset
from sklearn.utils import Bunch                          # For fallback data
from sklearn.model_selection import train_test_split     # Split data into train/test
from sklearn.preprocessing import StandardScaler         # Scale features to same range
from sklearn.linear_model import LinearRegression        # Model 1: Linear Regression
from sklearn.tree import DecisionTreeRegressor           # Model 2: Decision Tree
from sklearn.ensemble import RandomForestRegressor       # Model 3: Random Forest
from sklearn.metrics import mean_squared_error, r2_score # To measure how good our models are

# Create a folder to save our plots
os.makedirs("plots", exist_ok=True)

# Set a nice style for our charts
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ============================================================
# STEP 1: Load the Data
# ============================================================
# The California Housing dataset comes built into scikit-learn.
# It contains info about houses in different California districts.

print("=" * 60)
print("STEP 1: Loading the Data")
print("=" * 60)

# Load the dataset
try:
    housing = fetch_california_housing()
except Exception:
    # Fallback: generate synthetic California-like housing data
    print("  ⚠️ Could not download dataset. Generating synthetic data instead.")
    np.random.seed(42)
    n = 20640
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]
    data = np.column_stack([
        np.random.lognormal(1.2, 0.6, n),         # MedInc
        np.random.uniform(1, 52, n),                # HouseAge
        np.random.lognormal(1.6, 0.4, n),          # AveRooms
        np.random.lognormal(0.1, 0.3, n),          # AveBedrms
        np.random.lognormal(6.5, 1.0, n),          # Population
        np.random.lognormal(1.0, 0.5, n),          # AveOccup
        np.random.uniform(32.5, 42.0, n),           # Latitude
        np.random.uniform(-124.3, -114.3, n),       # Longitude
    ])
    target = (0.4 * data[:, 0] + 0.1 * data[:, 1] / 52 + 0.05 * data[:, 2]
              - 0.3 * data[:, 5] / 10 + np.random.normal(0, 0.3, n))
    target = np.clip(target, 0.15, 5.0)
    housing = Bunch(data=data, target=target, feature_names=feature_names)

# Convert it to a pandas DataFrame (a nice table format)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target  # This is what we want to predict (house price)

# Let's see what our data looks like
print(f"\n📊 Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n📋 Column names and what they mean:")
print("  - MedInc      → Median income in the district")
print("  - HouseAge    → Median age of houses")
print("  - AveRooms    → Average number of rooms per house")
print("  - AveBedrms   → Average number of bedrooms per house")
print("  - Population  → District population")
print("  - AveOccup    → Average number of people per house")
print("  - Latitude    → Location (north-south)")
print("  - Longitude   → Location (east-west)")
print("  - MedHouseVal → Median house value (TARGET - what we predict)")

print(f"\n🔍 First 5 rows of data:")
print(df.head())

print(f"\n📈 Basic statistics:")
print(df.describe().round(2))


# ============================================================
# STEP 2: Explore & Visualize the Data (EDA)
# ============================================================
# EDA = Exploratory Data Analysis
# We look at the data to find patterns before building models.

print("\n" + "=" * 60)
print("STEP 2: Exploring the Data (EDA)")
print("=" * 60)

# --- Visualization 1: Correlation Heatmap ---
# Correlation tells us how strongly two things are related.
# Values close to 1 or -1 = strong relationship
# Values close to 0 = weak relationship

print("\n📊 Creating correlation heatmap...")

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,          # Show numbers on the heatmap
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Color scheme (blue = negative, red = positive)
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()
print("  ✅ Saved: plots/correlation_heatmap.png")

# Let's see which features correlate most with house price
print("\n🔗 Correlation with House Price (MedHouseVal):")
correlations = correlation_matrix["MedHouseVal"].drop("MedHouseVal").sort_values(ascending=False)
for feature, corr in correlations.items():
    bar = "█" * int(abs(corr) * 20)
    sign = "+" if corr > 0 else "-"
    print(f"  {sign} {feature:15s}: {corr:+.3f}  {bar}")


# ============================================================
# STEP 3: Prepare the Data for Modeling
# ============================================================
# We need to split our data into:
#   - Training set (80%) → The model learns from this
#   - Testing set (20%)  → We check how good the model is on this

print("\n" + "=" * 60)
print("STEP 3: Preparing Data for Modeling")
print("=" * 60)

# Separate features (X) from target (y)
X = df.drop("MedHouseVal", axis=1)  # Everything except the price
y = df["MedHouseVal"]                # Just the price

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42      # Makes results reproducible
)

print(f"  📦 Training set: {X_train.shape[0]} samples")
print(f"  🧪 Testing set:  {X_test.shape[0]} samples")

# Scale the features (makes all features have similar ranges)
# This helps some models work better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  ✅ Features scaled using StandardScaler")


# ============================================================
# STEP 4: Train 3 Different Models
# ============================================================
# We'll try 3 models and see which one predicts best!

print("\n" + "=" * 60)
print("STEP 4: Training Models")
print("=" * 60)

# Dictionary to store our models and their results
results = {}

# --- Model 1: Linear Regression ---
# The simplest model. Draws a straight line through the data.
print("\n🔵 Model 1: Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)         # Train the model
lr_predictions = lr_model.predict(X_test_scaled)  # Make predictions

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)
results["Linear Regression"] = {"RMSE": lr_rmse, "R²": lr_r2, "predictions": lr_predictions}
print(f"  RMSE: {lr_rmse:.4f}  (lower is better)")
print(f"  R²:   {lr_r2:.4f}   (closer to 1 is better)")

# --- Model 2: Decision Tree ---
# Makes predictions by asking yes/no questions about the data.
print("\n🟢 Model 2: Decision Tree")
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_r2 = r2_score(y_test, dt_predictions)
results["Decision Tree"] = {"RMSE": dt_rmse, "R²": dt_r2, "predictions": dt_predictions}
print(f"  RMSE: {dt_rmse:.4f}")
print(f"  R²:   {dt_r2:.4f}")

# --- Model 3: Random Forest ---
# Combines many decision trees for better results. Usually the best!
print("\n🟠 Model 3: Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)
results["Random Forest"] = {"RMSE": rf_rmse, "R²": rf_r2, "predictions": rf_predictions}
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  R²:   {rf_r2:.4f}")


# ============================================================
# STEP 5: Compare Models & Create Visualizations
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Comparing Models")
print("=" * 60)

# Print comparison table
print("\n📊 Model Comparison:")
print(f"  {'Model':<25s} {'RMSE':<10s} {'R² Score':<10s}")
print("  " + "-" * 45)
best_model_name = None
best_r2 = -1
for name, metrics in results.items():
    marker = ""
    if metrics["R²"] > best_r2:
        best_r2 = metrics["R²"]
        best_model_name = name
    print(f"  {name:<25s} {metrics['RMSE']:<10.4f} {metrics['R²']:<10.4f}")

print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")


# --- Visualization 2: Actual vs Predicted (for best model) ---
print("\n📊 Creating Actual vs Predicted plot...")

best_predictions = results[best_model_name]["predictions"]
plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_predictions, alpha=0.3, s=10, color="#2196F3")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", linewidth=2, label="Perfect Prediction"
)
plt.xlabel("Actual Price", fontsize=13)
plt.ylabel("Predicted Price", fontsize=13)
plt.title(f"Actual vs Predicted House Prices\n({best_model_name} — R² = {best_r2:.4f})", fontsize=15, fontweight="bold")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=150)
plt.close()
print("  ✅ Saved: plots/actual_vs_predicted.png")


# --- Visualization 3: Feature Importance (Random Forest) ---
print("\n📊 Creating feature importance plot...")

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=housing.feature_names
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
feature_importance.plot(kind="barh", color=colors, edgecolor="black", linewidth=0.5)
plt.xlabel("Importance Score", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.title("Feature Importance (Random Forest)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("  ✅ Saved: plots/feature_importance.png")


# ============================================================
# DONE! 🎉
# ============================================================
print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print(f"\n🏆 Best model: {best_model_name}")
print(f"📊 R² Score: {best_r2:.4f} ({best_r2*100:.1f}% of price variation explained)")
print(f"📉 RMSE: {results[best_model_name]['RMSE']:.4f}")
print(f"\n📁 Check the 'plots/' folder for visualizations:")
print(f"   1. correlation_heatmap.png")
print(f"   2. actual_vs_predicted.png")
print(f"   3. feature_importance.png")
print(f"\n💡 Key Insight: Median Income is the strongest predictor of house prices!")
