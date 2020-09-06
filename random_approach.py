# random approach
# randomly guess whether the person survived or not

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# read the dataset
dataset = pd.read_csv("test.csv")

# survived array
survived_array = []

for i in range(892, 1310):
    survived = random.choice([0,1])
    survived_array.append(survived)

# write it to the gender submission
submission = pd.read_csv("gender_submission.csv")
submission['Survived'] = survived_array
submission.to_csv('submission.csv', index=False)


    
