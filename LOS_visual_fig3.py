import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Project 1/healthcare/train_data.csv')

df = df.drop_duplicates()

df.dropna(inplace=True)
df.reset_index(drop=True)

fig, axs = plt.subplots(3,2)
gynecology = df.loc[df.Department == 'gynecology'].loc[:,'Stay'].value_counts(normalize=True)
anesthesia = df.loc[df.Department == 'anesthesia'].loc[:,'Stay'].value_counts(normalize=True)
radiotherapy = df.loc[df.Department == 'radiotherapy'].loc[:,'Stay'].value_counts(normalize=True)
tb = df.loc[df.Department == 'TB & Chest disease'].loc[:,'Stay'].value_counts(normalize=True)
surgery = df.loc[df.Department == 'surgery'].loc[:,'Stay'].value_counts(normalize=True)

gyn = plt.subplot(3,2,1)
gyn.bar(gynecology.index, gynecology.values, color='r')
gyn.set_title('Gynecology Stay Length Frequencies')
# gyn.set_xlabel('Stay Lengths')
gyn.set_ylabel('Frequency')
anes = plt.subplot(3,2,2)
anes.bar(anesthesia.index, anesthesia.values, color='g')
anes.set_title('Anesthesia Stay Length Frequencies')
# anes.set_xlabel('Stay Lengths')
anes.set_ylabel('Frequency')
rad = plt.subplot(3,2,3)
rad.bar(radiotherapy.index , radiotherapy.values, color='y')
rad.set_title('Radiotherapy Stay Length Frequencies')
# rad.set_xlabel('Stay Lengths')
rad.set_ylabel('Frequency')
tbp = plt.subplot(3,2,4)
tbp.bar(tb.index, tb.values, color='b')
tbp.set_title('TB & Chest disease Stay Length Frequencies')
tbp.set_xlabel('Stay Lengths')
tbp.set_ylabel('Frequency')
sug = plt.subplot(3,2,5)
sug.bar(surgery.index, surgery.values, color=(1,.76,.01))
sug.set_title('Surgery Admission Stay Length Frequencies')
sug.set_xlabel('Stay Lengths')
sug.set_ylabel('Frequency')
sug.autoscale()
fig.delaxes(axs[2][1])


for ax in fig.axes:
    
    plt.sca(ax)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()