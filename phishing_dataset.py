import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load Dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Step 2: Check for missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Step 3: Drop duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
df = df.drop_duplicates()

# Step 4: Drop columns (simulate reduced feature model)
if 'label' in df.columns:
    selected_cols = [col for col in df.columns if col != 'label'][:3]
    df = df[selected_cols + ['label']]
else:
    raise ValueError("Missing 'label' column")

# Step 5: Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# Step 6: Handle missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Step 7: Display class distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='label')
plt.title("Class Distribution After Preprocessing")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Step 8: Split dataset into features and target
X = df.drop(columns=['label'])
y = df['label'].astype(int)

# Step 9: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 10: Train/Test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.5, stratify=y, random_state=42
)

# Step 11: Logistic Regression Model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Step 12: Evaluation
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 13: Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
