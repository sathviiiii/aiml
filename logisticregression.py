import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Step 1: Load the data
data = pd.read_csv('sample_data.csv')
# Step 2: Data Preprocessing
# Separate features (X) and target label (y)
X = data.drop('target', axis=1)
y = data['target']
# Handle missing values (if any) - optional step
X.fillna(X.mean(), inplace=True)
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 3: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)
# Step 4: Prediction
y_pred = model.predict(X_test)
# Step 5: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')