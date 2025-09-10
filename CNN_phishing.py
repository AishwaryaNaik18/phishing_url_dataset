import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("phishing_dataset_processed_cleaned.csv")

# Check if label column exists
target_column = 'label'
if target_column not in df.columns:
    raise ValueError(f"'{target_column}' not found in dataset columns.")

# Drop missing values (if any)
df.dropna(inplace=True)

# Encode categorical features
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# -------------------------
# 2. Split features & target
# -------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for Conv1D
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# One-hot encode labels if multi-class
if len(np.unique(y)) > 2:
    y_encoded = to_categorical(y)
    output_units = y_encoded.shape[1]
    output_activation = 'softmax'
    loss_function = 'categorical_crossentropy'
else:
    y_encoded = y.values
    output_units = 1
    output_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42
)

# -------------------------
# 3. Build CNN model
# -------------------------
model = Sequential()
model.add(Input(shape=(X_reshaped.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_units, activation=output_activation))

# Compile
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# -------------------------
# 4. Train model & store history
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------------------------
# 5. Evaluate on test set
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# 6. Classification Report & Confusion Matrix
# -------------------------
y_pred_probs = model.predict(X_test)

# If binary classification → threshold at 0.5
if output_units == 1:
    y_pred = (y_pred_probs > 0.5).astype("int32")
else:  # multi-class → argmax
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test = np.argmax(y_test, axis=1)

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
# 7. Plot training history
# -------------------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
