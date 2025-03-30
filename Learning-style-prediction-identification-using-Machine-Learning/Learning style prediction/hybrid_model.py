
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('scaled_X_features.csv')

# Separate features (X) and target variable (y)
X = df.drop('learning_style', axis=1)
y = df['learning_style']

# Apply SMOTE for balancing the dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize features
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define classifiers
svm_classifier = SVC(C=100, gamma=1, kernel='linear', probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
nn_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

# Create an ensemble voting classifier
voting_classifier = VotingClassifier(
    estimators=[('svm', svm_classifier), ('knn', knn_classifier), ('nn', nn_classifier)],
    voting='soft'
)

# Train the voting classifier
voting_classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = voting_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1_en = f1_score(y_test, y_pred, average="weighted")

print("Hybrid Model Accuracy:", accuracy)
print("F1 Score:", f1_en)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Cross-validation
cv_scores = cross_val_score(voting_classifier, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean accuracy: {:.2f} %".format(cv_scores.mean() * 100))
print("Standard Deviation: {:.2f} %".format(cv_scores.std() * 100))

# Classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model
joblib.dump(voting_classifier, 'hybrid_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Hybrid model and scaler saved successfully.")
