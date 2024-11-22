from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
# Load the dataset
df = pd.read_csv("ssi_classification_data.csv")

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=["Procedure_Name", "Gender", "Diabetes_Status", "Wound_Class"], drop_first=True)

# Define features and target
X = df_encoded.drop(columns=["Patient_ID", "SSI_Type"])
y = df["SSI_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(class_weight='balanced', random_state=42) 
model.fit(X_train, y_train)
# Save the trained model using joblib
# Save the column names used during training
training_columns = list(X.columns)
joblib.dump(training_columns, 'training_columns.joblib')

joblib.dump(model, 'iris_model.joblib')
