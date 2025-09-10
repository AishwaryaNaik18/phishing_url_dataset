import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("phishing_cleaned_dataset.csv")

# Drop unused or non-numeric columns
df = df.drop(columns=["FILENAME", "URL", "Domain", "Title"], errors="ignore")

# Split into features and label
X = df.drop(columns=["label"])
y = df["label"]

# -------------------------
# 2. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Train Random Forest
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 4. Predictions
# -------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For ROC-AUC

# -------------------------
# 5. Evaluation Metrics
# -------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n Model Performance Metrics")
print(f"Accuracy   : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -------------------------
# 6. SHAP Explainability
# -------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle binary classification (class 1 SHAP values)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Class 1 = phishing
else:
    shap_vals = shap_values[..., 1] if shap_values.ndim == 3 else shap_values

# Mean absolute SHAP values for each feature
shap_importance = np.abs(shap_vals).mean(axis=0)

# Create DataFrame and sort
shap_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": shap_importance
}).sort_values(by="Importance", ascending=True)

# Plot Top 15 Features
plt.figure(figsize=(8, 6))
plt.barh(shap_df["Feature"].tail(15), shap_df["Importance"].tail(15), color="skyblue")
plt.title("Top 15 Important Features (SHAP)")
plt.xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig("shap_top_features.png")
plt.show()

# SHAP Summary Plot (Bee Swarm)
shap.summary_plot(shap_vals, X_test, feature_names=X_test.columns)
