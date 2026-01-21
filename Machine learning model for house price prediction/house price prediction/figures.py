from joblib import load
import matplotlib.pyplot as plt
import numpy as np



model = load('house_price_model.joblib')
# Make sure these exist from your earlier code:

# y_price_test, preds, y_price_train, train_preds

# mae, rmse, r2, train_mae, train_r2

# model  (Pipeline with "prep" and "xgb" steps)


# For convenience:

y_test_actual = np.array(y_price_test)

y_train_actual = np.array(y_price_train)

y_test_pred = np.array(preds)

y_train_pred = np.array(train_preds)

# -------------------------------------------------------------------

# 1. Predicted vs Actual (Test set)

# -------------------------------------------------------------------

plt.figure()

plt.scatter(y_test_actual, y_test_pred, alpha=0.3)

max_val = max(y_test_actual.max(), y_test_pred.max())

min_val = min(y_test_actual.min(), y_test_pred.min())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual price (£)")

plt.ylabel("Predicted price (£)")

plt.title("Predicted vs Actual House Prices (Test set)")

plt.tight_layout()

plt.show()

# -------------------------------------------------------------------

# 2. Histogram of residuals (Test set)

# -------------------------------------------------------------------

residuals = y_test_actual - y_test_pred

plt.figure()

plt.hist(residuals, bins=50)

plt.xlabel("Error = Actual − Predicted (£)")

plt.ylabel("Count")

plt.title("Distribution of Prediction Errors (Test set)")

plt.tight_layout()

plt.show()

# -------------------------------------------------------------------

# 3. Residuals vs Predicted (Test set)

# -------------------------------------------------------------------

plt.figure()

plt.scatter(y_test_pred, residuals, alpha=0.3)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted price (£)")

plt.ylabel("Residual (Actual − Predicted)")

plt.title("Residuals vs Predicted Values (Test set)")

plt.tight_layout()

plt.show()

# -------------------------------------------------------------------

# 4. Feature importance (XGBoost inside the Pipeline)

# -------------------------------------------------------------------

# Get feature names from the ColumnTransformer

prep = model.named_steps["prep"]

xgb = model.named_steps["xgb"]

feature_names = prep.get_feature_names_out()

importances = xgb.feature_importances_

# Sort and take top 20

idx = np.argsort(importances)[::-1][:20]

top_features = feature_names[idx]

top_importances = importances[idx]

plt.figure()

plt.barh(range(len(top_features)), top_importances[::-1])

plt.yticks(range(len(top_features)), top_features[::-1])

plt.xlabel("Importance")

plt.title("Top 20 Feature Importances (XGBoost)")

plt.tight_layout()

plt.show()

# -------------------------------------------------------------------

# 5. Train vs Test metrics (MAE and R²)

# -------------------------------------------------------------------

metrics = ["MAE", "R²"]

train_values = [train_mae, train_r2]

test_values = [mae, r2]

x = np.arange(len(metrics))

width = 0.35

plt.figure()

plt.bar(x - width / 2, train_values, width, label="Train")

plt.bar(x + width / 2, test_values, width, label="Test")

plt.xticks(x, metrics)

plt.ylabel("Score")

plt.title("Train vs Test Performance")

plt.legend()

plt.tight_layout()

plt.show()