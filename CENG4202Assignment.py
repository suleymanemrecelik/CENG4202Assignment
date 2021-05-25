#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Import required libraries

from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Read dataset

df = pd.read_csv('diabetes.csv')
df


# In[50]:


#pregnancies	Pregnancies	Number of pregnancies
#gender	Gender	F: Female, M: Male
#glucose	Glucose	Glucose level
#bloodPressure	BloodPressure	Blood pressure value
#skinThickness	SkinThickness	Skin thickness value
#insulin	Insulin	Insulin level
#bmi	BMI	Body mass index
#age	Age	Age
#outcome	Outcome	0: Not diabetic, 1: Diabetic
#calorieIntake	CalorieIntake	Number of calories consumed
#exercise	Exercise	No: No exercise, Both: Morning and evening exercises, Evening: Evening exercise, Morning: Morning exercise
#sleepDuration	SleepDuration	Sleep time in hours


# In[51]:


#Find columns with NaN values

df.isnull().sum()


# In[52]:


#Fill NaN values with median of CalorieIntake column

calorieIntake = df.iloc[:, 10]
calorieIntake.fillna(calorieIntake.median(skipna=True), inplace=True)


# In[53]:


#Fill 0 values with mean of the columns.

bloodPressure = df.iloc[:, 3]
bloodPressure.replace(0, bloodPressure.mean(), inplace=True)

skinThickness = df.iloc[:, 4]
skinThickness.replace(0, skinThickness.mean(), inplace=True)

insulin = df.iloc[:, 5]
insulin.replace(0, insulin.mean(), inplace=True)


# In[54]:


#Convert char values to integer in Gender column 
#0 for F
#1 for M

gender = df.iloc[:, 1]
gender_encoder = LabelEncoder()
gender = gender_encoder.fit_transform(gender)
df.iloc[:, 1] = gender


# In[55]:


#Convert string values to integers in Exercise column
#0 for Both
#1 for Evening
#2 for Morning
#3 for No

exercise = df.iloc[:, 11]
exercise_encoder = LabelEncoder()
exercise = exercise_encoder.fit_transform(exercise)
df.iloc[:, 11] = exercise
df


# In[56]:


#Feature selection, importance value of each column is the output

x = df.iloc[:, [0,1,2,3,4,5,6,7,8,10,11,12]]
y = df.iloc[:,9]


# In[57]:


#Scale data with StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# In[58]:


#Splitting the data into test data and train data

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=0.25, random_state=1)


# In[59]:


#Generate classification model with Support Vector Machine 

modelSVM = SVC(kernel='rbf', random_state=0, gamma='auto')
modelSVM.fit(x_tr,y_tr.values.ravel())


# In[60]:


#Get accuracy score of Support Vector Machine

predictSVM = modelSVM.predict(x_ts)
print("Accuracy: " + str(accuracy_score(y_ts, predictSVM) * 100) + '%')
print("\nConfusion matrix: ") 
print(confusion_matrix(y_ts, predictSVM))
print("\nEvaluation scores: ")
print(classification_report(y_ts, predictSVM))


# In[61]:


#Generate classification model with Random Forest Classifier

modelRFC=RandomForestClassifier(random_state=0, n_estimators=100)
modelRFC.fit(x_tr,y_tr.values.ravel())


# In[62]:


#Get accuracy score and feature importances of Random Forest Classifier

predictRFC = modelRFC.predict(x_ts)

print("Accuracy: " + str(accuracy_score(y_ts, predictRFC) * 100) + '%')
print("\nFeature importances: ")
print(modelRFC.feature_importances_)

feature_names = ['Pregnancies','Gender','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','CalorieIntake','Exercise','SleepDuration']
std = np.std([tree.feature_importances_ for tree in modelRFC.estimators_], axis=0)
indices = np.argsort(modelRFC.feature_importances_)
plt.figure()
plt.title("Feature Importances")
plt.barh(range(x.shape[1]), modelRFC.feature_importances_[indices],
       color="b")
plt.yticks(range(x.shape[1]), [feature_names[i] for i in indices])
plt.ylim([-1, x.shape[1]])
plt.show()

print("\nConfusion matrix: ") 
print(confusion_matrix(y_ts, predictRFC))
print("\nEvaluation scores: ")
print(classification_report(y_ts, predictRFC))


# In[63]:


#Get parameters of Random Forest Classifier to do hyperparameter tuning

modelRFC.get_params().keys()


# In[16]:


#Use GridSearchCV to get best parameters

params = {"n_estimators" : [500, 600, 750, 1000],
          "max_depth" : [6, 5, 4, 3],
          "n_jobs" : [1, 2, 3]         
          }


modelCV = GridSearchCV(modelRFC,
                       params,
                       cv=10,
                       n_jobs=-1,
                       verbose=2).fit(x_tr,y_tr.values.ravel())


# In[17]:


#Get best parameters as a result of GridSearchCV

modelCV.best_params_


# In[22]:


#Execute Random Forest Classifier with better parameters to get higher accuracy score

modelTuned = RandomForestClassifier(max_depth=6,
                                    n_estimators=500,
                                    n_jobs=1)

modelTuned.fit(x_tr,y_tr.values.ravel())
predict_survival2 = modelTuned.predict(x_ts)
print("Accuracy: " + str(accuracy_score(y_ts, predict_survival2) * 100) + '%')

