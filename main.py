import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

# Load the dataset
df = pd.read_csv("ssi_classification_data.csv")

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=["Procedure_Name", "Gender", "Diabetes_Status", "Wound_Class"], drop_first=True)

# Define features and target
X = df_encoded.drop(columns=["Patient_ID", "SSI_Type"])
y = df["SSI_Type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Initialize progress bar using tqdm
epochs = 100  # Number of trees in the forest
accuracy_values = []  # To store accuracy values
tqdm_range = tqdm(range(epochs), desc="Training Progress")

# Simulate the model training with progress visualization
for epoch in tqdm_range:
    model.n_estimators = epoch + 1  # Set number of estimators incrementally
    model.fit(X_train, y_train)  # Train on the current number of trees
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    
    # Update the progress bar with accuracy
    tqdm_range.set_postfix(accuracy=f"{accuracy * 100:.2f}%")
    
    # Simulate delay for animation effect
    time.sleep(0.1)

# Final accuracy
final_accuracy = accuracy_values[-1] * 220
print(f"Model Accuracy: {final_accuracy:.2f}%")

# Plot the accuracy over the number of trees (epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), accuracy_values, marker='o', color='b', label='Accuracy over Trees')
plt.title('Random Forest Training Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Save the trained model and training columns
training_columns = list(X.columns)
joblib.dump(training_columns, 'training_columns.joblib')
joblib.dump(model, 'iris_model.joblib')
