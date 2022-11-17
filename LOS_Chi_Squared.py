import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

df = pd.read_csv('Project 1/healthcare/train_data.csv')

df = df.drop_duplicates()

df.dropna(inplace=True)
df.reset_index(drop=True)

cross = pd.crosstab(index=df["Type of Admission"] , columns=df["Stay"], margins=True)

df = df.loc[:,['Available Extra Rooms in Hospital' , 'Stay']]

print(df)
label_encoder = LabelEncoder()
df['Available Extra Rooms in Hospital'] = label_encoder.fit_transform(df['Available Extra Rooms in Hospital'])
df['Stay'] = label_encoder.fit_transform(df['Stay'])
print(df)

X = df.drop('Stay', axis=1)
print(X)
Y = df['Stay']
print(Y)

chi_scores = chi2(X,Y)
print(chi_scores)