import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")

#Column names
data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

#Discard the data points that contain missing values
data = data.replace(['?'], np.nan)
data = data.dropna()

#changes the num column values. If the value is 0 then it will be 0. numbers 1-4 are changed to 1
data.num = [0 if i == 0 else 1 for i in data.num]

# Extract the target class
classData = data.iloc[:,13]

# Extract the data attributes
attributeData = data.iloc[:, 0:13]

# Split the data for training and for testing
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, test_size = 0.2, random_state = 2)

# Create decision tree classifier
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)

#Train the classifier using the training attributes and class
clf = clf.fit(dataTrain, classTrain)

# Apply the decision tree to classify the test data
predC = clf.predict(dataTest)

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is', accuracy_score(classTest, predC))

#print tree
tree.plot_tree(clf, fontsize = 10)





