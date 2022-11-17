import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Project 1/healthcare/train_data.csv')
# df = pd.read_csv('healthcare/train_data.csv')

df = df.drop_duplicates()

df.dropna(inplace=True)
df.reset_index(drop=True)


stay_counts = df.loc[:,'Stay'].value_counts(normalize=True)

print(stay_counts)

plt.bar(stay_counts.index , stay_counts.values, color=(1,.76,.01))
plt.xlabel("Length of Stay")
plt.ylabel("Frequency")
plt.title('Frequencies of Cases with each Stay Length')
plt.show()



