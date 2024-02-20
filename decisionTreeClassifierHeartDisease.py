import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, auc

# Read data
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")

# Column names
data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Discard the data points that contain missing values
data = data.replace(['?'], np.nan)
data = data.dropna()

# Changes the num column values. If the value is 0, then it will be 0. Numbers 1-4 are changed to 1
data.num = [0 if i == 0 else 1 for i in data.num]

# Extract the target class
classData = data.iloc[:, 13]

# Extract the data attributes
attributeData = data.iloc[:, 0:13]

# Normalize the data attributes
scaler = MinMaxScaler()
attributeData_normalized = scaler.fit_transform(attributeData)
# Split the data for training and for testing
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData_normalized, classData, test_size=0.2, random_state=2)

# Create decision tree classifier
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Train the classifier using the training attributes and class
clf = clf.fit(dataTrain, classTrain)

# Apply the decision tree to classify the test data
predC = clf.predict(dataTest)

# More detailed classification report
print("Classification Report:")
print(classification_report(classTest, predC))

# Plot tree
plt.figure(figsize=(12, 6))
tree.plot_tree(clf, fontsize=10, filled=True, feature_names=attributeData.columns, class_names=['No Disease', 'Disease'])
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(classTest, clf.predict_proba(dataTest)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
