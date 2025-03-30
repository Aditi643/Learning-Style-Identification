import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Read the dataset
df = pd.read_csv('scaled_X_features.csv')  # Ensure the file exists

# Separate features (X) and target variable (y)
X = df.drop('learning_style', axis=1)
y = df['learning_style']

# Apply SMOTE for balancing the dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
svm_classifier = SVC(C=100, gamma=1, kernel='linear')
rf_classifier = RandomForestClassifier(
    max_depth=None, max_features='sqrt', min_samples_leaf=1,
    min_samples_split=2, n_estimators=300, random_state=0
)

# Create an ensemble voting classifier
voting_classifier = VotingClassifier(
    estimators=[('svm', svm_classifier), ('rf', rf_classifier)],
    voting='hard'
)

# Train the voting classifier
voting_classifier.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = voting_classifier.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1_en = f1_score(y_test, y_pred, average="weighted")

print("Ensemble Model Accuracy:", accuracy)
print("F1 Score:", f1_en)

# Confusion matrix display
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=voting_classifier.classes_)
disp.plot()

# Cross-validation
cv_scores = cross_val_score(voting_classifier, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean accuracy: {:.2f} %".format(cv_scores.mean() * 100))
print("Standard Deviation: {:.2f} %".format(cv_scores.std() * 100))

# Classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model and scaler
joblib.dump(voting_classifier, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")

# ----------- Load and Use the Model for Prediction -----------
def predict_learning_style():
    """
    Function to take user input, scale features using the saved scaler,
    load the model, and predict the learning style.
    """
    try:
        # Load the saved model and scaler
        classifier = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Define feature names
        feature_names = [
            'Amount of time spent by learners interacting with images',
            'Amount of time spent on video related materials',
            'Amount of time spent on text-based material',
            'Amount of time spent on audio-related materials',
            'Complexity or depth of the learning material',
            'Frequency of PowerPoint usage',
            'Concrete contents',
            'Performance or achievement of the learner',
            'Number of correctly answered standard questions',
            'Number of messages or posts posted by the learner',
            'Time or duration spent by the learner in solving exercises',
            'Number of group discussions',
            'Number of lessons of learning objectives skipped',
            'Number of times the learner utilized the Next button',
            'Amount of Time Spent in sessions',
            'Number of questions on topics',
            'Number of questions or queries posed by the learner'
        ]

        # Take user input for features
        input_features = []
        for feature in feature_names:
            value = float(input(f"Enter value for {feature}: "))
            input_features.append(value)

        # Convert input features to a NumPy array and scale them
        input_features = np.array(input_features).reshape(1, -1)
        scaled_input_features = scaler.transform(input_features)

        # Predict the target label
        target_label = classifier.predict(scaled_input_features)

        # Map numeric labels to their respective learning styles
        label_mapping = {0: 'Processing', 1: 'Understanding', 2: 'Input', 3: 'Perception'}
        predicted_label = label_mapping[target_label[0]]

        print("Predicted Learning Style:", predicted_label)
    
    except Exception as e:
        print("Error:", str(e))

