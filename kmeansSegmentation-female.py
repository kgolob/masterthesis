import datetime
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

mandant = 'xxxlutz_de'
sampleSize = '250k'
#path = '/media/backup/MasterThesis/output'
path = '/home/kgolob/Repos/masterthesis/out/kmeans_female_output'
clusters = 15
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


def printToFile(string) :
    print(string, file=outputFile)
    return


def printClusterToFiles(data, file) :
    clusterOutputFile = open(file, 'w')
    print(data, file=clusterOutputFile, end='\n\n')
    clusterOutputFile.close()
    return


# load data from csv
#df = pd.read_csv('/media/backup/MasterThesis/customerData_{}_{}.csv'.format(mandant, sampleSize), delimiter=';')
df = pd.read_csv('/home/kgolob/Repos/masterthesis/customerData_{}_{}.csv'.format(mandant, sampleSize), delimiter=';')

# drop customers which do not have a gender assigned - they do not provide any value for segmentation
df = df.dropna(subset=['Gender'])
# drop customers with negative orderSum - could have happened by a bug in the shop
df.drop(df[ (df['TotalOrderSum'] < 0) ].index, inplace=True)
# drop company customers - its only 1 in 250k customers and this one has no orders or anything else of value
df.drop(df[ (df['Gender'] == 'COMPANY') ].index, inplace=True)
# drop male gender, to get female only data
df.drop(df[ (df['Gender'] == 'MALE') ].index, inplace=True)

# drop not relevant columns
df_metrics = df.drop(columns=['Id', 'City', 'PostalCode'])

# replace all NA cells with 0
df_metrics = df_metrics.fillna(0)

# create category for females of age between 20-35
df_metrics['Age_categorical'] = 'FALSE'
df_metrics.loc[(df_metrics['Age'] >= 20) & (df_metrics['Age'] <= 35), 'Age_categorical'] = 'TRUE'
# df_metrics.loc[(df_metrics['Age'] >= 0) & (df_metrics['Age'] < 40), 'Age_categorical'] = 'FALSE'

# make a copy of data frame
df_tr = df_metrics

# transform to dummies
df_tr = pd.get_dummies(df_tr, columns=['Gender',  'BonusCardOwner', 'Age_categorical'])
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
clmns = ['OrderCount', 'TotalOrderSum', 'ReservationCount', 'Age_categorical_FALSE', 'Age_categorical_TRUE', 'BonusCardOwner_False', 'BonusCardOwner_True']
df_tr_std = stats.zscore(df_tr[clmns])

printToFile("Compute kmeans clustering...")
st = time.time()

# cluster the data
kmeans = KMeans(n_clusters=clusters, random_state=0, init='k-means++', n_jobs=-1).fit(df_tr_std)
labels = kmeans.labels_

elapsed_time = time.time() - st
printToFile("Elapsed time: %.2fs" % elapsed_time)
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

# reattach original ids to dataframe
df_tr['Id'] = df['Id']

# save dataframe to file
df_tr.to_csv(path_or_buf= outputPath +'clustered_customerData_{}_{}.csv'.format(mandant, sampleSize), sep=';')

outputFile.close()

####################   Visualisation   ####################
plt.rc('font', size=16)
plt.rc('axes', labelsize=30)
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
# plt.figure(figsize=(12, 12))
# plt.scatter(df_tr.iloc[:,1], df_tr.iloc[:, 2],  c=kmeans.labels_.astype(float))
# plt.xlim(0, 40)
# plt.xlabel('Alter')
# plt.ylim(0, 10000)
# plt.ylabel('Total spent in €')
# savePlot(plt, 'testplot')

####################   Histograms   ####################
# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Alter')
plt.ylabel('Anzahl')
n, bins, patches = plt.hist(df_tr['Age'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Alter', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_Age')

# plt.figure(figsize=(12, 12))
# f, (ax1, ax2) = plt.subplots(2)
# arr = np.random.normal(10, 3, size=1000)
# cnts, bins = np.histogram(df_tr['Age'], bins=10, range=(df_tr['Age'].min(), df_tr['Age'].max()))
# ax1.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))
# ax2.hist(df_tr['Age'], bins=10, range=(df_tr['Age'].min(), df_tr['Age'].max()))
# savePlot(plt, 'Numpy_Histogram_Age')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Anzahl der Bestellungen')
plt.ylabel('Anzahl')
n, bins, patches = plt.hist(df_tr['OrderCount'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Anzahl der Bestellungen', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_OrderCount')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Anzahl der Bestellungen')
plt.ylabel('Anzahl')
plt.ylim(0, 100)
n, bins, patches = plt.hist(df_tr['OrderCount'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Anzahl der Bestellungen', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_OrderCount_zoomed')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Ausgaben in €')
plt.ylabel('Anzahl')
n, bins, patches = plt.hist(df_tr['TotalOrderSum'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Ausgaben in €', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_TotalOrderSum')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Ausgaben in €')
plt.xticks(np.arange(0, 225000, 25000))
plt.ylabel('Anzahl')
plt.ylim(0, 40)
n, bins, patches = plt.hist(df_tr['TotalOrderSum'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Ausgaben in €', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_TotalOrderSum_zoomed')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Anzahl der Reservierungen')
plt.ylabel('Anzahl')
n, bins, patches = plt.hist(df_tr['ReservationCount'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Anzahl der Reservierungen', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_ReservationCount')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Anzahl der Reservierungen')
#plt.xticks(np.arange(0, 225000, 25000))
plt.ylabel('Anzahl')
plt.ylim(0, 2500)
n, bins, patches = plt.hist(df_tr['ReservationCount'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Anzahl der Reservierungen', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_ReservationCount_zoomed')

cm = plt.cm.get_cmap('RdYlBu_r')
plt.figure(figsize=(12, 12))
plt.xlabel('Anzahl der Reservierungen')
#plt.xticks(np.arange(0, 225000, 25000))
plt.ylabel('Anzahl')
plt.ylim(0, 500)
n, bins, patches = plt.hist(df_tr['ReservationCount'], bins=50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.title('Histogramm Anzahl der Reservierungen', fontdict={'fontsize': 30})
savePlot(plt, 'Histogram_ReservationCount_zoomed2')
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

####################   2D/3D-Plots   ####################
plt.rc('axes', labelsize=30)
fig = plt.figure(figsize=(16,16))
# ax = fig.add_subplot(111)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=210)
x = np.array(df_tr.iloc[:,0])
y = np.array(df_tr.iloc[:,1])
z = np.array(df_tr.iloc[:,2])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Alter', labelpad=15)
ax.set_ylabel('Anzahl der Bestellungen', labelpad=15)
ax.set_zlabel('Ausgaben in €', labelpad=25)
# ax.axes.set_ylim3d(bottom=0, top=10)
# ax.axes.set_zlim3d(bottom=0, top=2500)
# plt.gca().invert_yaxis()
ax.scatter(x,y,z, c=kmeans.labels_.astype(float))
# savePlot(plt, '3d_plot')
savePlot(plt, '3d_plot_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))

plt.rc('axes', labelsize=30)
fig = plt.figure(figsize=(16,16))
# ax = fig.add_subplot(111)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=210)
x = np.array(df_tr.iloc[:,0])
y = np.array(df_tr.iloc[:,1])
z = np.array(df_tr.iloc[:,2])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Alter', labelpad=15)
ax.set_ylabel('Anzahl der Bestellungen', labelpad=15)
ax.set_zlabel('Ausgaben in €', labelpad=25)
ax.axes.set_ylim3d(bottom=0, top=10)
ax.axes.set_zlim3d(bottom=0, top=2500)
# plt.gca().invert_yaxis()
ax.scatter(x,y,z, c=kmeans.labels_.astype(float))
# savePlot(plt, '3d_plot')
savePlot(plt, '3d_plot_zoomed_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))


####################   PCA-Plots   ####################
# PCA
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(df_tr)
print(X_reduced)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=kmeans.labels_.astype(float))
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=df_tr,
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])
savePlot(plt, 'PCA_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))


fig = plt.figure(figsize=(12, 12))
X_reduced = PCA(n_components=2).fit_transform(df_tr)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_.astype(float))
plt.ylim(0, 10000)
savePlot(plt, 'PCA_2d_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
