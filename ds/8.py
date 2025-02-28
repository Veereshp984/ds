import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.decomposition import PCA 

# Load the Iris dataset
iris = datasets.load_iris() 
x = iris.data 
y = iris.target 

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Create the SVM classifier
svm_classifier = SVC(kernel='linear', C=5.0, random_state=42) 
svm_classifier.fit(x_train, y_train) 

# Predict the class labels on the test data
y_pred = svm_classifier.predict(x_test) 

# Print confusion matrix
print("Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred)) 

# Print classification report
print("\nClassification Report:") 
print(classification_report(y_test, y_pred)) 

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")