# Heart Disease Classification
Completed data mining classification projects using Python. 
A dataset was pulled from UCI Machine Learning Repository [1]. The data was taken from 303 patients from a Cleveland Medical Centre, 137 of which have some form of heart disease rated on a scale from one to four, with zero representing no presence. The data set contains 14 medically related attributes used to predict the presence of heart disease.
Used classifiers such as: K nearesrt Neighbours, Random Forest, Decision Tree and Support vector machine using Sklearn packages. 
Used Pandas package to pre-process data. 

[1] Robert Detrano, M.D., Ph.D.. (1988). Heart Disease Databases [Cleveland]. V.A. Medical 
  Center, Long Beach and Cleveland Clinic Foundation.


Notes on the data
- The data was preprocessed by first removing any values with '?'
- The target column, lablled 'num' was originally labled with values ranging from 0 - 4, 0 being no signs of heart diease, and the rest being varying degrees of the diease, so a relabeling was 
preformed, and all values 1 - 4 were processed to the value 1, indicating the precence of heart diease. 

* copy and paste the reports here * 
talk about how the supports are different and that could be the cause of fluctuations in the imbalanced datasets. 