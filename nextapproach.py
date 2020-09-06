# 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# read the dataset
dataset = pd.read_csv("train.csv")

# save the mean of age
dataset["Age"] = dataset["Age"].replace(np.NaN, dataset["Age"].mean())
dataset["Embarked"] = dataset["Embarked"].replace(np.NaN, "R")          # replace it with letter R so better for categorical encoding later for better prediction
dataset["Cabin"] = dataset["Cabin"].replace(np.NaN, "Z")

print(dataset.isnull().sum())

