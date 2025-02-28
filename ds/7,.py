# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the dataset from '7.csv'
df = pd.read_csv("C:\\Users\\HP\\Desktop\\sev.csv")

# Display the first few rows of the dataset to verify
print("Dataset preview:")
print(df.head())

# Define the feature columns and the target column
feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                     'DiabetesPedigreeFunction', 'Age']
predicted_class_names = ['Outcome']  # The column representing diabetes outcome (0 or 1)

# Extract the features (X) and the target variable (y)
X = df[feature_col_names].values  # Features used for prediction
y = df[predicted_class_names].values  # The target variable (whether or not the person has diabetes)

# Splitting the dataset into training and testing data (33% test data)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Printing the number of training and test samples
print('\nTotal number of Training Data:', ytrain.shape)
print('Total number of Test Data:', ytest.shape)

# Training the Naive Bayes classifier on the training data
clf = GaussianNB().fit(xtrain, ytrain.ravel())

# Predicting the class labels on the test data
predicted = clf.predict(xtest)

# Predicting for a single individual with given feature values
predictTestData = clf.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Printing the performance metrics
print('\nConfusion Matrix:')
print(metrics.confusion_matrix(ytest, predicted))

print('\nAccuracy of the classifier:', metrics.accuracy_score(ytest, predicted))

print('Precision of the classifier:', metrics.precision_score(ytest, predicted))
print('Recall of the classifier:', metrics.recall_score(ytest, predicted))

# Predicted value for the individual test data
print("Predicted Value for individual Test Data:", predictTestData)