import numpy as np
import pandas as pd
df = pd.read_csv('/media/backup/MasterThesis/customerData_xxxlutz_de_100k.csv', delimiter=';')
csv_matrix = df.as_matrix()
print(df.shape)
print('_______________')
print(df.describe())
print('_______________')
print(df.head())
print('_______________')
print(list(df.columns))
print('_______________')
print(csv_matrix)