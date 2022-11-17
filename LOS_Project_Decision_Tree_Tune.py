import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('Project 1/healthcare/train_data.csv')



df = df.drop_duplicates()

df.dropna(inplace=True)
df.reset_index(drop=True)





# Mappings for categorical attributes to numeric attributes



Hospital_type_code_mappings = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6
}

Hospital_region_code_mappings = {
    'X' : 0,
    'Y' : 1,
    'Z' : 2
}

Department_mappings = {
    'gynecology' : 0,
    'anesthesia' : 1,
    'radiotherapy' : 2,
    'TB & Chest disease' : 3,
    'surgery' : 4
}

Ward_type_mappings = {
    'R' : 0,
    'Q' : 1,
    'S' : 2,
    'P' : 3,
    'T' : 4,
    'U' : 5,
}

Ward_facility_code_mappings = {
    'F' : 0,
    'E' : 1,
    'D' : 2,
    'C' : 3,
    'B' : 4,
    'A' : 5,
}

Type_of_admission_mappings = {
    'Trauma' : 0,
    'Emergency' : 1,
    'Urgent' : 2
}

Severity_of_illness_mappings = {
    'Moderate' : 0,
    'Minor' : 1,
    'Extreme' : 2
}

Age_mappings = {
    '0-10' : 0,
    '11-20' : 1,
    '21-30' : 2,
    '31-40' : 3,
    '41-50' : 4,
    '51-60' : 5,
    '61-70' : 6,
    '71-80' : 7,
    '81-90' : 8,
    '91-100' : 9
}

Stay_mappings = {
    '0-10' : 0,
    '11-20' : 1,
    '21-30' : 2,
    '31-40' : 3,
    '41-50' : 4,
    '51-60' : 5,
    '61-70' : 6,
    '71-80' : 7,
    '81-90' : 8,
    '91-100' : 9,
    'More than 100 Days' : 10
}

# using mappings on dataset

df['Hospital_type_code'] = df['Hospital_type_code'].map(Hospital_type_code_mappings)
df['Hospital_region_code'] = df['Hospital_region_code'].map(Hospital_region_code_mappings)
df['Department'] = df['Department'].map(Department_mappings)
df['Ward_Type'] = df['Ward_Type'].map(Ward_type_mappings)
df['Ward_Facility_Code'] = df['Ward_Facility_Code'].map(Ward_facility_code_mappings)
df['Type of Admission'] = df['Type of Admission'].map(Type_of_admission_mappings)
df['Severity of Illness'] = df['Severity of Illness'].map(Severity_of_illness_mappings)
df['Age'] = df['Age'].map(Age_mappings)
df['Stay'] = df['Stay'].map(Stay_mappings)



df['number_of_time_patient_visited']=df.groupby('patientid')['patientid'].transform('count')
df["Number_of_hospital_visits"] = df.groupby("patientid")['patientid'].rank(method="first", ascending=True)

df = df.drop('case_id', axis=1)
df = df.drop('patientid', axis=1)
df = df.drop('number_of_time_patient_visited', axis=1)





X = df.drop('Stay', axis=1)
y = df['Stay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=1)

for i in range(3,30):
    for j in ['gini' , 'entropy']:
        dtc = DecisionTreeClassifier(criterion=j, max_depth=i)
        dtc = dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        print("============================================")
        print(f'Max Depth: {i}')
        print(f'Criterion: {j}')
        print("============================================")
        print(f"classes: {dtc.n_classes_}")
        print(f'Depth: {dtc.get_depth()}')
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
        print("============================================\n")
