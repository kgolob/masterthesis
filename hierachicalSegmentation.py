import os
import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

mandant = 'xxxlutz_de'
sampleSize = '50k'
#path = '/media/backup/MasterThesis/hierachical_output'
path = '/home/kgolob/Repos/masterthesis/out/hierachical_output'
clusters = 9
# dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
dt = 'test'
outputPath = '{}_{}_{}_{}-clusters/'.format(path, mandant, sampleSize, clusters)
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
outputFile = open(outputPath + 'out.txt', 'w')
# set output options
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 999999)
pd.set_option('display.width', 1000)


def savePlot(plt, name):
    if not os.path.exists(outputPath + 'plots'):
        os.makedirs(outputPath + 'plots')
    plt.savefig(outputPath + 'plots/' + name)
    return


def printToFile(string):
    print(string, file=outputFile)
    return


def printClusterToFiles(data, file):
    clusterOutputFile = open(file, 'w')
    print(data, file=clusterOutputFile, end='\n\n')
    clusterOutputFile.close()
    return


# load data from csv
#df = pd.read_csv('/media/backup/MasterThesis/customerData_{}_{}.csv'.format(mandant, sampleSize), delimiter=';')
df = pd.read_csv('/home/kgolob/Repos/masterthesis/customerData_{}_{}.csv'.format(mandant, sampleSize), delimiter=';')

# drop customers which do not have a gender assigned - they do not provide any value for segmentation
df = df.dropna(subset=['Gender'])
# drop customers with negative orderSum - could have happend by a bug in the shop
df.drop(df[ (df['TotalOrderSum'] < 0) ].index, inplace=True)
# drop company customers - its only 1 in 250k customers and this one has no orders or anything else of value
df.drop(df[ (df['Gender'] == 'COMPANY') ].index, inplace=True)

# drop not relevant columns
df_metrics = df.drop(columns=['Id', 'City', 'PostalCode'])

# replace all NA cells with 0
df_metrics = df_metrics.fillna(0)

# make a copy of data frame
df_tr = df_metrics

# transform to dummies
df_tr = pd.get_dummies(df_tr, columns=['Gender',  'BonusCardOwner'])
printToFile(df_tr['Age'].max())
printToFile('#################################################')
printToFile(df_tr.describe())
printToFile('#################################################')
printToFile(df_tr.shape)
printToFile('#################################################')
printToFile(df_tr.head())
printToFile('#################################################')
printToFile(list(df_tr.columns))
printToFile('#################################################')

# standardize
# clmns = ['Age', 'OrderCount', 'TotalOrderSum', 'ReservationCount', 'Gender_FEMALE', 'Gender_MALE', 'BonusCardOwner_False', 'BonusCardOwner_True']
# df_tr_std = stats.zscore(df_tr[clmns])
# sns.clustermap(df_tr)

data_scaled = normalize(df_tr)
data_scaled = pd.DataFrame(data_scaled, columns=df_tr.columns)
print(data_scaled.head())


# plt.figure(figsize=(10, 7))
# plt.title("Dendrograms")
# dend = dendrogram(linkage(data_scaled, method='ward'))
# savePlot(plt, 'dendrogram')
#
# plt.figure(figsize=(10, 7))
# plt.title("Dendrograms")
# dend = dendrogram(linkage(data_scaled, method='ward'))
# plt.axhline(y=6, color='r', linestyle='--')
# savePlot(plt, 'dendrogram_clusters')

clmns = ['Age', 'OrderCount', 'TotalOrderSum', 'ReservationCount', 'Gender_FEMALE', 'Gender_MALE', 'BonusCardOwner_False', 'BonusCardOwner_True']

printToFile("Compute unstructured hierarchical clustering...")
st = time.time()
cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)

elapsed_time = time.time() - st
printToFile("Elapsed time: %.2fs" % elapsed_time)

labels = cluster.labels_

# glue back to original data
df_tr['clusters'] = labels

# add the column into our list
clmns.extend(['clusters'])

# lets analyze the clusters
printToFile('Cluster means')
printToFile(df_tr[clmns].groupby(['clusters']).mean())
printToFile('#################################################')
printToFile('Cluster min')
printToFile(df_tr[clmns].groupby(['clusters']).min())
printToFile('#################################################')
printToFile('Cluster max')
printToFile(df_tr[clmns].groupby(['clusters']).max())
printToFile('#################################################')
printToFile('Cluster std')
printToFile(df_tr[clmns].groupby(['clusters']).std())
printToFile('#################################################')
printToFile('describe')
printToFile(df_tr[clmns].groupby(['clusters']).describe())
printToFile('#################################################')

#################################
# printToFile all rows inside of clusters
grouped_df = df_tr.groupby('clusters')
if not os.path.exists(outputPath + 'cluster'):
    os.makedirs(outputPath + 'cluster')
else:
    shutil.rmtree(outputPath + 'cluster')
    os.makedirs(outputPath + 'cluster')

for key, item in grouped_df:
    printClusterToFiles(grouped_df.get_group(key), "{}cluster/cluster{}.txt".format(outputPath, key))
#####################################
outputFile.flush()

plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['OrderCount'], data_scaled['TotalOrderSum'], c=cluster.labels_)
plt.xlabel('OrderCount')
plt.ylabel('TotalOrderSum')
savePlot(plt, '2d')


plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Age'], data_scaled['TotalOrderSum'], c=cluster.labels_)
plt.xlabel('Age')
plt.ylabel('TotalOrderSum')
savePlot(plt, '2d_2')
# reduced_data = PCA(n_components=2).fit_transform(df_tr_std)
# linked = linkage(reduced_data, 'single')
#
# labelList = range(1, 11)
#
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='top',
#             labels=labelList,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()

# cluster the data
# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(df_tr_std)
# labels = kmeans.labels_
#
# # glue back to original data
# df_tr['clusters'] = labels
#
# # add the column into our list
# clmns.extend(['clusters'])
#
# # lets analyze the clusters
# printToFile(df_tr[clmns].groupby(['clusters']).mean())
# printToFile('#################################################')
#
# #################################
# # printToFile all rows inside of clusters
# grouped_df = df_tr.groupby('clusters')
# if not os.path.exists(outputPath + 'cluster'):
#     os.makedirs(outputPath + 'cluster')
# else:
#     shutil.rmtree(outputPath + 'cluster')
#     os.makedirs(outputPath + 'cluster')
#
# for key, item in grouped_df:
#     printClusterToFiles(grouped_df.get_group(key), "{}cluster/cluster{}.txt".format(outputPath, key))
# #####################################
#
# # reattach original ids to dataframe
# df_tr['Id'] = df['Id']
#
# # save dataframe to file
# df_tr.to_csv(path_or_buf= outputPath +'clustered_customerData_{}_{}.csv'.format(mandant, sampleSize), sep=';')
#
# outputFile.close()
