import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the CSV files
train_csv = "E:/Coding Ninjas Frontend/CSS/chars74k_train.csv"  # Path to your train CSV file
test_csv = "E:/Coding Ninjas Frontend/CSS/chars74k_test.csv"    # Path to your test CSV file

# Load data into pandas DataFrames
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Step 2: Separate features (X) and labels (y)
X_train = train_data.iloc[:, 1:].values  # Pixel values (features)
y_train = train_data.iloc[:, 0].values   # Labels (characters)

X_test = test_data.iloc[:, 1:].values    # Pixel values (features)
y_test = test_data.iloc[:, 0].values     # Labels (characters)

# Step 3: Encode the labels (if they are characters)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 4: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)

# Step 5: Make predictions on the test set
y_pred_encoded = rf_model.predict(X_test)

# Decode the predicted labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
