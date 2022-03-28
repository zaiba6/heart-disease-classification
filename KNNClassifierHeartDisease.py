import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")

data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# #Discard the data points that contain missing values
data = data.replace(['?'], np.nan)
data = data.dropna()

data.num = [0 if i == 0 else 1 for i in data.num]

# Extract the data attributes//attributes
attibute = data.iloc[:, 0:13]

# Extract the target class//traget
target = data.iloc[:,13]

attibute_train, attibute_test, target_train, target_test = train_test_split(attibute, target, test_size = 0.33)

# #Create KNN Classifier
near = KNeighborsClassifier(n_neighbors = 3)

# #Train the model 
near.fit(attibute_train, target_train)

# #Predict the response for test dataset
target_pred = near.predict(attibute_test)

print('The accuracy of the classifier is', accuracy_score(target_test, target_pred))