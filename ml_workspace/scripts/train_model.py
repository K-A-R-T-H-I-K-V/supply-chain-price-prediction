from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Project root and paths
PROJECT_ROOT = Path().resolve()
DATA_PATH = PROJECT_ROOT / 'data' / 'superstoreSales_utf8.csv'
NOTEBOOKS_PATH = PROJECT_ROOT / 'notebooks'
os.makedirs(NOTEBOOKS_PATH, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print("Dataset loaded successfully with UTF-8 encoding!")
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found.")
    print("Ensure 'superstoreSales_utf8.csv' is in 'data/' folder.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# === Preprocessing ===
print("\n=== Preprocessing ===")

# 1. Drop irrelevant columns and potential leakage
drop_cols = ['Row ID', 'Order ID', 'Customer Name', 'Product Name', 'Province', 'Region', 'Customer Segment', 'Product Sub-Category', 'Product Container', 'Ship Date', 'Sales', 'Profit']
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

# 2. Handle missing values
print("\nMissing Values Before Imputation:")
print(df.isnull().sum())
df['Product Base Margin'] = df['Product Base Margin'].fillna(df['Product Base Margin'].median())
print("\nImputed Product Base Margin with median:", df['Product Base Margin'].median())
print("Missing Values After Imputation:")
print(df.isnull().sum())

# 3. Feature engineering: Extract seasonality features
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['IsHoliday'] = df['Month'].isin([11, 12]).astype(int)
df = df.drop(columns=['Order Date'])
print("\nExtracted seasonality features: Month, Quarter, IsHoliday")

# 4. Handle outliers
unit_price_mean = 90.76  # From EDA
unit_price_std = 62.0    # From EDA
unit_price_cap = unit_price_mean + 2 * unit_price_std
df['Unit Price'] = df['Unit Price'].clip(upper=unit_price_cap)
print(f"Capped Unit Price outliers at ${unit_price_cap:.2f}")

shipping_cost_mean = df['Shipping Cost'].mean()
shipping_cost_std = df['Shipping Cost'].std()
shipping_cost_cap = shipping_cost_mean + 2 * shipping_cost_std
df['Shipping Cost'] = df['Shipping Cost'].clip(upper=shipping_cost_cap)
print(f"Capped Shipping Cost outliers at ${shipping_cost_cap:.2f}")

# 5. Transform skewed features
df['Unit Price Log'] = np.log1p(df['Unit Price'])
df['Shipping Cost Log'] = np.log1p(df['Shipping Cost'])
print("\nApplied log-transform to Unit Price and Shipping Cost")

# 6. Additional feature engineering
# Bin Discount into categories to reduce cardinality
bins = [0.0, 0.05, 0.1, 0.25]  # Bins: low (0-0.05), medium (0.05-0.1), high (0.1-0.25)
labels = ['low', 'medium', 'high']
df['Discount Bin'] = pd.cut(df['Discount'], bins=bins, labels=labels, include_lowest=True)
# Create interaction term with binned Discount
df['Discount_Category'] = df['Discount Bin'].astype(str) + '_' + df['Product Category']
print("Added binned interaction term: Discount_Category")

# 7. Verify data
print("\nData Summary After Preprocessing:")
print(df.describe())
print("\nData Types:")
print(df.dtypes)

# Define features and target
X = df.drop(columns=['Unit Price', 'Unit Price Log', 'Discount Bin'])
y = df['Unit Price Log']

# Define categorical and numerical columns
categorical_cols = ['Order Priority', 'Ship Mode', 'Product Category', 'Discount_Category']
numerical_cols = ['Order Quantity', 'Discount', 'Shipping Cost Log', 'Product Base Margin', 'Month', 'Quarter', 'IsHoliday']

# === Model Training ===
# Preprocessing pipeline with scaling for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

# --- Random Forest Model ---
# Create pipeline with Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# Initial Random Forest training
rf_pipeline.fit(X_train, y_train)
y_pred_log_rf = rf_pipeline.predict(X_test)

# Inverse transform predictions
y_test_original = np.expm1(y_test)
y_pred_original_rf = np.expm1(y_pred_log_rf)

# Evaluate initial Random Forest model
rmse_rf = root_mean_squared_error(y_test_original, y_pred_original_rf)
r2_rf = r2_score(y_test_original, y_pred_original_rf)
print("\n=== Initial Random Forest Performance ===")
print(f"RMSE (Original Scale): ${rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")

# Hyperparameter tuning for Random Forest with RandomizedSearchCV
rf_param_dist = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__max_depth': [10, 20, 30, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}
rf_random_search = RandomizedSearchCV(rf_pipeline, param_distributions=rf_param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=2, random_state=42)
rf_random_search.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = rf_random_search.best_estimator_
y_pred_log_best_rf = best_rf_model.predict(X_test)
y_pred_best_original_rf = np.expm1(y_pred_log_best_rf)

# Evaluate tuned Random Forest model
rmse_best_rf = root_mean_squared_error(y_test_original, y_pred_best_original_rf)
r2_best_rf = r2_score(y_test_original, y_pred_best_original_rf)
print("\n=== Tuned Random Forest Performance ===")
print(f"Best Parameters: {rf_random_search.best_params_}")
print(f"Tuned RMSE (Original Scale): ${rmse_best_rf:.2f}")
print(f"Tuned R² Score: {r2_best_rf:.2f}")

# --- XGBoost Model ---
# Create pipeline with XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
])

# Initial XGBoost training
xgb_pipeline.fit(X_train, y_train)
y_pred_log_xgb = xgb_pipeline.predict(X_test)

# Inverse transform predictions
y_pred_original_xgb = np.expm1(y_pred_log_xgb)

# Evaluate initial XGBoost model
rmse_xgb = root_mean_squared_error(y_test_original, y_pred_original_xgb)
r2_xgb = r2_score(y_test_original, y_pred_original_xgb)
print("\n=== Initial XGBoost Performance ===")
print(f"RMSE (Original Scale): ${rmse_xgb:.2f}")
print(f"R² Score: {r2_xgb:.2f}")

# Hyperparameter tuning for XGBoost with RandomizedSearchCV
xgb_param_dist = {
    'regressor__n_estimators': [300, 500, 1000],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__reg_alpha': [0, 0.1, 1.0],
    'regressor__reg_lambda': [0, 1.0, 10.0]
}
xgb_random_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=xgb_param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=2, random_state=42
)
xgb_random_search.fit(X_train, y_train)

# Best XGBoost model
best_xgb_pipeline = xgb_random_search.best_estimator_
y_pred_log_best_xgb = best_xgb_pipeline.predict(X_test)
y_pred_best_original_xgb = np.expm1(y_pred_log_best_xgb)

# Evaluate tuned XGBoost model
rmse_best_xgb = root_mean_squared_error(y_test_original, y_pred_best_original_xgb)
r2_best_xgb = r2_score(y_test_original, y_pred_best_original_xgb)
print("\n=== Tuned XGBoost Performance ===")
print(f"Best Parameters: {xgb_random_search.best_params_}")
print(f"Tuned RMSE (Original Scale): ${rmse_best_xgb:.2f}")
print(f"Tuned R² Score: {r2_best_xgb:.2f}")

# Select the best model (Random Forest or XGBoost) based on RMSE
if rmse_best_xgb < rmse_best_rf:
    print("\nXGBoost performed better than Random Forest. Using XGBoost model for final predictions and visualizations.")
    best_model = best_xgb_pipeline
    y_pred_best_original = y_pred_best_original_xgb
    rmse_best = rmse_best_xgb
    r2_best = r2_best_xgb
else:
    print("\nRandom Forest performed better than or equal to XGBoost. Using Random Forest model for final predictions and visualizations.")
    best_model = best_rf_model
    y_pred_best_original = y_pred_best_original_rf
    rmse_best = rmse_best_rf
    r2_best = r2_best_rf

# Feature importance
cat_feature_names = best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_cols)
feature_names = list(cat_feature_names) + numerical_cols
importances = best_model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title(f'Feature Importance for Unit Price Prediction ({best_model.named_steps["regressor"].__class__.__name__})')
plt.savefig(str(NOTEBOOKS_PATH / 'feature_importance.png'))
plt.close()

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_best_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual Unit Price')
plt.ylabel('Predicted Unit Price')
plt.title(f'Actual vs Predicted Unit Price ({best_model.named_steps["regressor"].__class__.__name__})')
plt.savefig(str(NOTEBOOKS_PATH / 'actual_vs_predicted.png'))
plt.close()

# Save predictions for blockchain logging (Step 3)
test_predictions = pd.DataFrame({
    'Actual Unit Price': y_test_original,
    'Predicted Unit Price': y_pred_best_original
})
test_predictions.to_csv(PROJECT_ROOT / 'data' / 'test_predictions.csv', index=False)
print("\nTest predictions saved to 'data/test_predictions.csv' for blockchain logging")

print("\nStep 2 completed! Model trained and evaluated.")
print("Visualizations saved in 'notebooks/' folder:")
print("- feature_importance.png")
print("- actual_vs_predicted.png")
print("Next steps:")
print("- Review feature importance to understand key price drivers")
print("- Check actual vs predicted plot for model fit")
print("- Proceed to Step 3 for blockchain integration")