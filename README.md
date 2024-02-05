# Heart Disease Classification

This repository contains a completed data mining classification project implemented in Python. The project focus on predicting the presence of heart disease using machine learning techniques. 

## Dataset
The dataset used in these projects was sourced from the UCI Machine Learning Repository [1]. It consists of data collected from 303 patients at a Cleveland Medical Centre. Among these patients, 137 have been diagnosed with some form of heart disease, rated on a scale from one to four, with zero representing no presence of heart disease. The dataset comprises 14 medically related attributes that are utilized to predict the presence or absence of heart disease.

[1] Robert Detrano, M.D., Ph.D.. (1988). Heart Disease Databases [Cleveland]. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation.

### Data Preprocessing
- Removing any instances with missing values ('?'). 
- The target column, labeled 'num', originally contained values ranging from 0 to 4. All values from 1 to 4 were consolidated to 1, indicating the presence of heart disease.

## Implemented Classifiers
Various classifiers were employed in this project using the scikit-learn (sklearn) package:
- K Nearest Neighbors
- Random Forest
- Decision Tree
- Support Vector Machine (This file includes graphs illustrating attribute distributions)

### Classification Reports
    -----------------------------------------------------------------
                            K Nearest Neighbors
    -----------------------------------------------------------------
                  precision    recall  f1-score   support

               0       0.84      0.75      0.79        56
               1       0.71      0.81      0.76        42

        accuracy                           0.78        98
       macro avg       0.77      0.78      0.77        98
    weighted avg       0.78      0.78      0.78        98

    -----------------------------------------------------------------
                                Random Forest
    -----------------------------------------------------------------
                  precision    recall  f1-score   support

               0       0.73      0.85      0.79        48
               1       0.84      0.71      0.77        51

        accuracy                           0.78        99
       macro avg       0.78      0.78      0.78        99
    weighted avg       0.79      0.78      0.78        99

    -----------------------------------------------------------------
                              Decision Tree
    -----------------------------------------------------------------
                  precision    recall  f1-score   support

               0       0.88      0.86      0.87        35
               1       0.81      0.84      0.82        25

        accuracy                           0.85        60
       macro avg       0.85      0.85      0.85        60
    weighted avg       0.85      0.85      0.85        60

    -----------------------------------------------------------------
                          Support Vector Machine
    -----------------------------------------------------------------
                  precision    recall  f1-score   support

               0       0.86      0.86      0.86        51
               1       0.82      0.82      0.82        39

        accuracy                           0.84        90
       macro avg       0.84      0.84      0.84        90
    weighted avg       0.84      0.84      0.84        90


### Note on Support Counts
It's observed that the support counts for different classes vary, indicating potential class imbalance within the dataset. Fluctuations in support counts could lead to imbalanced datasets, which may affect the performance of the classifiers. Further investigation into class distribution and potential remedies for class imbalance is warranted.