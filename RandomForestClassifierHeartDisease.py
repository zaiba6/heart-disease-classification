import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# Read the data
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = None)

data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

#Discard the data points that contain missing values
data = data.replace(['?'], np.nan)
data = data.dropna()

data.num = [0 if i == 0 else 1 for i in data.num]

# Extract the data attributes//attributes
attributes = data.iloc[:, 0:13]

# Extract the target class//traget
target = data.iloc[:,13]

scaler = MinMaxScaler()
attributeData_normalized = scaler.fit_transform(attributes)

#split into training and testing
attributes_train, attributes_test, target_train, target_test = train_test_split(attributeData_normalized, target, test_size = 0.33)

#call the classifier and give estimation number
gaussC = RandomForestClassifier(n_estimators = 100)

#Find the fit using classifier
gaussC.fit(attributes_train, target_train)

#to predict accuracy
targ_pred = gaussC.predict(attributes_test)

# More detailed classification report
print(classification_report(target_test, targ_pred))