import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Step 1: Load the Dataset
df = pd.read_csv('sample_data.csv')
# Step 2: Preprocess the Data
# Assuming the last column is the target variable and the rest are features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Train the Naïve Bayesian Classifier
model = GaussianNB()
model.fit(X_train, y_train)
# Step 5: Make Predictions
y_pred = model.predict(X_test)
# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
# Output the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")