import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

# Load Iris dataset
iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
y = pd.DataFrame(iris.target)
y.columns = ['targets']

# KMeans model
model = KMeans(n_clusters=3)
model.fit(x)

# Plotting the real classification and KMeans classification
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 2, 1)
plt.scatter(x.petal_length, x.petal_width, c=colormap[y.targets], s=40)
plt.title('Real Classification')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.subplot(1, 2, 2)
plt.scatter(x.petal_length, x.petal_width, c=colormap[model.labels_], s=40)
plt.title('KMeans Classification')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Calculate accuracy and confusion matrix for KMeans
print('Accuracy score of KMeans:', sm.accuracy_score(y, model.labels_))
print('Confusion matrix of KMeans:', sm.confusion_matrix(y, model.labels_))

# Standardize the data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns=x.columns)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)

y_gmm = gmm.predict(xs)

# Plotting GMM classification
plt.subplot(2, 2, 3)
plt.scatter(x.petal_length, x.petal_width, c=colormap[y_gmm], s=40)
plt.title('GMM Classification')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Calculate accuracy and confusion matrix for GMM
print('Accuracy score of EM:', sm.accuracy_score(y, y_gmm))
print('Confusion matrix of EM:', sm.confusion_matrix(y, y_gmm))

plt.show()