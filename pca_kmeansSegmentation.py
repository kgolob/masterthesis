import datetime
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

mandant = 'xxxlutz_de'
sampleSize = '250k'
path = '/media/backup/MasterThesis/pca_output'
clusters = 12
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

reduced_data = PCA(n_components=2).fit_transform(df_tr_std)
# kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
kmeans = KMeans(n_clusters=clusters, random_state=0, init='k-means++', n_jobs=-1)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
scatter = plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the customer dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
handles = [plt.Line2D([], [], marker="o", ls="",
                      color=scatter.cmap(scatter.norm(yi))) for yi in centroids]
#plt.legend((line1, line2, line3), ('label1', 'label2', 'label3'))
#plt.legend(handles, np.unique(kmeans.labels_)
# plt.show()
savePlot(plt, 'pca_kmeans_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))

# cluster the data
# kmeans = KMeans(n_clusters=clusters, random_state=0, init='k-means++', n_jobs=-1).fit(df_tr_std)
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
