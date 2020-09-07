# random forest

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
# read the dataset
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# handle missing values
# print(df_train.isnull().sum())          # age, cabin, and embarked
# print("\n\n\n")
# print(df_test.isnull().sum())           # age, fare and cabin


df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())

df_train["Embarked"] = df_train["Embarked"].fillna('X')
# cabin in test dataset
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())

# label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df_train["Sex"] = encoder.fit_transform(df_train["Sex"])
df_test["Sex"] = encoder.fit_transform(df_test["Sex"])

# make independent and dependent variables
X_train = df_train[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
y_train = df_train["Survived"]
X_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
# prepare the model
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='gini', max_depth=4, max_leaf_nodes=5, max_samples=None)
# fit the data
classifier.fit(X_train, y_train)
# predict
y_pred = classifier.predict(X_test)


# submission
submission = pd.read_csv("gender_submission.csv")
submission["Survived"] = y_pred
submission.to_csv("submission.csv", index=False)


print("Success")

