from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import datasets

# Load the dataset
iris = datasets.load_iris() 
x = iris.data 
y = iris.target 

# Print feature names and data
print('Features: sepal-length, sepal-width, petal-length, petal-width') 
print(x) 

# Print target names and data
print('Classes: 0-Iris-Setosa, 1- Iris-Versicolour, 2- Iris-Virginica') 
print(y) 

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 

# Create the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5) 

# Train the classifier
classifier.fit(x_train, y_train) 

# Make predictions
y_pred = classifier.predict(x_test) 

# Print the confusion matrix
print('Confusion Matrix') 
print(confusion_matrix(y_test, y_pred)) 

# Print accuracy metrics
print('Accuracy Metrics') 
print(classification_report(y_test, y_pred))