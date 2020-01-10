import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

plt.figure(figsize=(12, 12))

# load data from csv
df = pd.read_csv('/media/backup/MasterThesis/customerData_xxxlutz_de_100k.csv', delimiter=';')
# drop not relevant columns
df = df.drop(columns=['Id', 'City', 'PostalCode'])

# drop customers which do not have a gender assigned - they do not provide any value for segmentation
df = df.dropna(subset=['Gender'])

# replace all NA cells with 0
df = df.fillna(0)

csv_matrix = df.as_matrix()
# make a copy of data frame
df_tr = df

# transform to dummies
df_tr = pd.get_dummies(df_tr, columns=['Gender',  'BonusCardOwner'])

print(df_tr.describe())
print('#################################################')
print(df_tr.shape)
print('#################################################')
print(df_tr.head())
print('#################################################')
print(list(df_tr.columns))
print('#################################################')

# standardize
clmns = ['Age', 'OrderCount', 'TotalOrderSum', 'ReservationCount', 'Gender_FEMALE', 'Gender_MALE', 'BonusCardOwner_False', 'BonusCardOwner_True']
df_tr_std = stats.zscore(df_tr[clmns])

# cluster the data
kmeans = KMeans(n_clusters=15, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

# glue back to original data
df_tr['clusters'] = labels

# add the column into our list
clmns.extend(['clusters'])

# lets analyze the clusters
print(df_tr[clmns].groupby(['clusters']).mean())
print('#################################################')
# print all rows inside of clusters
grouped_df = df_tr.groupby('clusters')

for key, item in grouped_df:
    print(grouped_df.get_group(key), "\n\n")

df_tr.to_csv(path_or_buf='/media/backup/MasterThesis/clustered_customerData_xxxlutz_de_100k.csv', sep=';')


################ elbow method to find best number of clusters ###########
# k means determine k
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_tr_std)
    kmeanModel.fit(df_tr_std)
    distortions.append(sum(np.min(cdist(df_tr_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_tr_std.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
################ end elbow method to find best number of clusters ###########
