import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df.dropna(inplace=True)

# Drop 'Person ID' if it exists
if 'Person ID' in df.columns:
    df.drop(columns=['Person ID'], inplace=True)

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Print class distribution (debug)
print("Class distribution:\n", df['Sleep Disorder'].value_counts())

# Features and target
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with class balancing
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy on test set: {accuracy:.2f}")

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump((model, label_encoders, X.columns.tolist()), f)

print("✅ Model trained and saved to rf_model.pkl")
