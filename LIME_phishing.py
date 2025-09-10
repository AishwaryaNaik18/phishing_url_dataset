import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("phishing_cleaned_dataset.csv")

# Drop unused or non-numeric columns
df = df.drop(columns=["FILENAME", "URL", "Domain", "Title"], errors="ignore")

# Features and label
X = df.drop(columns=["label"])
y = df["label"]

# -------------------------
# 2. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Fit Random Forest
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 4. Evaluate model
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# 5. LIME explanation for a sample instance
# -------------------------
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["Legitimate", "Phishing"],
    mode='classification'
)

# Choose one instance to explain
i = 5  # You can change this index
exp = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=model.predict_proba,
    num_features=10
)

# Text explanation in console
print("\nExplanation for instance", i)
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# Save HTML explanation
exp.save_to_file("lime_explanation_instance_5.html")

# Optional: matplotlib plot
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
