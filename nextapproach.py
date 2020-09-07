# 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# read the dataset
dataset = pd.read_csv("train.csv")

# save the mean of age
dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])          # replace it with letter R so better for categorical encoding later for better prediction

# converting categorical columns into numerical columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for data in dataset:
    dataset["Sex"] = encoder.fit_transform(dataset["Sex"])
    dataset["Embarked"] = encoder.fit_transform(dataset["Embarked"])
    dataset["Age"] = dataset["Age"].astype("int64")

# check each category by percentage of survived passengers
target = ['Survived']
selected = ["Sex", "Pclass", "Embarked", "SibSp", "Parch", "Age"]

# for x in selected:
#     print("Survival Percentage By: ", x)
#     print(dataset[[x, target[0]]].groupby(x, as_index=False).mean(), '\n')


# visualization
import seaborn as sns

plt.figure(figsize=(10, 5))
plt.hist(x=[dataset[dataset["Survived"] == 1] ['Age'], dataset[dataset["Survived"]==0]['Age']], stacked=True, color=['b', 'r'], label=["Survived", "Dead"])
plt.title("Survived By Age")
plt.xlabel("Age")
plt.ylabel("No of Passengers")
plt.legend()
plt.show()


# model 
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(bootstrap=True,
                                    ccp_alpha=0.0,
                                    class_weight=None,
                                    criterion='gini', max_depth=4, max_features='auto', max_leaf_nodes=5,
                                    max_samples=None)

X_train = dataset.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y_train = dataset["Survived"]

classifier.fit(X_train, y_train)

# predict
df_test = pd.read_csv("test.csv")

df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])
# convert 
for data in df_test:
    df_test["Sex"] = encoder.fit_transform(df_test["Sex"])
    df_test["Embarked"] = encoder.fit_transform(df_test["Embarked"])
    df_test["Age"] = df_test["Age"].astype("int64")
X_test = df_test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)



y_pred = classifier.predict(X_test)
