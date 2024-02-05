import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = None)

data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal','num']

#Discard the data points that contain missing values
data = data.replace(['?'], np.nan)
data = data.dropna()

#changing the target values 1 - 4 to just 1
data.num = [0 if i == 0 else 1 for i in data.num]

# # Plot each histogram separately
# for column in data.columns:
#     plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
#     data[column].hist()
#     plt.title(column)  # Set title as the column name
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.show()

# Extract the target class// y 
classData = data.iloc[:,13]

# Extract the data attributes//x
attributeData = data.iloc[:, 0:13]

# Normalize the data attributes
scaler = MinMaxScaler()
attributeData_normalized = scaler.fit_transform(attributeData)

# Split the data for training and for testing
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData_normalized, classData, test_size = 0.3, random_state = 1)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(dataTrain, classTrain)

#Predict the response for test dataset
y_pred = clf.predict(dataTest)

# More detailed classification report
print(classification_report(classTest, y_pred))