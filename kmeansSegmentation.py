import os
import datetime
import shutil

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

mandant = 'xxxlutz_at'
sampleSize = '100k'
path = '/media/backup/MasterThesis/output'
clusters = 18
# dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
dt = 'test'
outputPath = '{}_{}_{}_{}-clusters/'.format(path, mandant, sampleSize, clusters)
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
outputFile = open(outputPath + 'out.txt', 'w')
# set output options
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 999999)
pd.set_option('display.width', 1000)


def savePlot(plt, name):
    if not os.path.exists(outputPath + 'plots'):
        os.makedirs(outputPath + 'plots')
    plt.savefig(outputPath + 'plots/' + name)
    return


def printToFile(string) :
    print(string, file=outputFile)
    return


def printClusterToFiles(data, file) :
    clusterOutputFile = open(file, 'w')
    print(data, file=clusterOutputFile, end='\n\n')
    clusterOutputFile.close()
    return


# load data from csv
df = pd.read_csv('/media/backup/MasterThesis/customerData_{}_{}.csv'.format(mandant, sampleSize), delimiter=';')

# drop customers which do not have a gender assigned - they do not provide any value for segmentation
df = df.dropna(subset=['Gender'])

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
clmns = ['Age', 'OrderCount', 'TotalOrderSum', 'ReservationCount', 'Gender_FEMALE', 'Gender_MALE', 'BonusCardOwner_False', 'BonusCardOwner_True']
df_tr_std = stats.zscore(df_tr[clmns])

# cluster the data
kmeans = KMeans(n_clusters=clusters, random_state=0, init='k-means++', n_jobs=-1).fit(df_tr_std)
labels = kmeans.labels_

# glue back to original data
df_tr['clusters'] = labels

# add the column into our list
clmns.extend(['clusters'])

# lets analyze the clusters
printToFile(df_tr[clmns].groupby(['clusters']).mean())
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

# reattach original ids to dataframe
df_tr['Id'] = df['Id']

# save dataframe to file
df_tr.to_csv(path_or_buf= outputPath +'clustered_customerData_{}_{}.csv'.format(mandant, sampleSize), sep=';')

outputFile.close()

###################################################################
# Visualize it:
# plt.subplot(221)
# plt.xlabel('TotalOrderSum')
# plt.ylabel('clusters')
# plt.scatter(df_tr.iloc['TotalOrderSum'], df_tr['clusters'],  c=kmeans.labels_.astype(float))
# plt.subplot(222)
# plt.xlabel('Age')
# plt.ylabel('clusters')
# plt.scatter(df_tr['Age'], df_tr['clusters'],  c=kmeans.labels_.astype(float))
# plt.subplot(223)
# plt.xlabel('Gender_Male')
# plt.ylabel('clusters')
# plt.scatter(df_tr['Gender_MALE'], df_tr['clusters'],  c=kmeans.labels_.astype(float))
plt.figure(figsize=(12, 12))
plt.scatter(df_tr.iloc[:,1], df_tr.iloc[:, 2],  c=kmeans.labels_.astype(float))
plt.xlim(0, 40)
plt.xlabel('Age')
plt.ylim(0, 10000)
plt.ylabel('Total spent in €')
savePlot(plt, 'testplot')
#plt.show()





################ elbow method to find best number of clusters ###########
# k means determine k
# distortions = []
# K = range(1, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(df_tr_std)
#     kmeanModel.fit(df_tr_std)
#     distortions.append(sum(np.min(cdist(df_tr_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_tr_std.shape[0])
#
# # Plot the elbow
# plt.figure(figsize=(12, 12))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# # plt.show()
# savePlot(plt, 'elbow_method')
################ end elbow method to find best number of clusters ###########


fig = plt.figure(figsize=(16,16))
# ax = fig.add_subplot(111)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=210)
x = np.array(df_tr.iloc[:,0])
y = np.array(df_tr.iloc[:,1])
z = np.array(df_tr.iloc[:,2])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Age')
ax.set_ylabel('Number of orders')
ax.set_zlabel('Total spent in €')
# ax.axes.set_ylim3d(bottom=0, top=10)
# ax.axes.set_zlim3d(bottom=0, top=2500)
# plt.gca().invert_yaxis()
ax.scatter(x,y,z, c=kmeans.labels_.astype(float))
# savePlot(plt, '3d_plot')
savePlot(plt, '3d_plot_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))

fig = plt.figure(figsize=(16,16))
# ax = fig.add_subplot(111)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=210)
x = np.array(df_tr.iloc[:,0])
y = np.array(df_tr.iloc[:,1])
z = np.array(df_tr.iloc[:,2])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Age')
ax.set_ylabel('Number of orders')
ax.set_zlabel('Total spent in €')
ax.axes.set_ylim3d(bottom=0, top=10)
ax.axes.set_zlim3d(bottom=0, top=2500)
# plt.gca().invert_yaxis()
ax.scatter(x,y,z, c=kmeans.labels_.astype(float))
# savePlot(plt, '3d_plot')
savePlot(plt, '3d_plot_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))







