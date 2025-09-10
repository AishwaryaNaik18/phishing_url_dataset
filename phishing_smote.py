import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# Step 1: Load Dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Step 2: Encode categorical features
label_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Step 3: Handle missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Step 4: Drop duplicates
df.drop_duplicates(inplace=True)

# Step 5: Define target
target_col = 'label'
if target_col not in df.columns:
    raise ValueError("Target column 'label' not found.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Step 6: Visualize original class distribution
print("\n--- Original Class Distribution ---")
print(Counter(y))
sns.countplot(x=y)
plt.title("Original Class Distribution")
plt.show()

# Step 7: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Apply SMOTE
print("\n✅ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\n--- After SMOTE ---")
print(Counter(y_resampled))
sns.countplot(x=y_resampled)
plt.title("Balanced Class Distribution (After SMOTE)")
plt.show()

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Step 10: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 11: Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 12: Confusion Matrix Plot
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', square=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

