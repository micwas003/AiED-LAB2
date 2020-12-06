from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn as skl
import plotly.figure_factory as ff
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from heatmapcluster import heatmapcluster


n = input("Please provide number of clusters (default 5): ") or 5


def readSalesData(path):
    return pd.read_csv(path)

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def kMeansClustering(data, n):
    kMeans = KMeans(n_clusters=n, init='random')
    kMeans.fit(data)
    kMeansPP = KMeans(n_clusters=n, init='k-means++')
    kMeansPP.fit(data)
    return kMeans.labels_, kMeansPP.labels_

def plotSelectedData(data, colX, colY, title, labels):
    coloursDict = {1: "red", 2: "blue", 3: "green"}
    plt.scatter(data[colX], data[colY], c=labels)
    plt.title(title)
    plt.xlabel(colX)
    plt.ylabel(colY)
    plt.show()

def agglomerativeClustering(data, n):
    agglomerative = AgglomerativeClustering(n_clusters=n)
    agglomerative.fit(data)
    print(agglomerative.labels_)
    return agglomerative

def dbscanClustering(data):
    dbscan = DBSCAN()
    dbscan.fit(data)
    return dbscan.labels_

data = readSalesData('dataset/Sales_Transactions_Dataset_Weekly.csv')
filteredData = data[['Normalized 0','Normalized 1', 'Normalized 2', 'Normalized 3', 'Normalized 4', 'Normalized 5', 'Normalized 6', 'Normalized 7', 'Normalized 8', 'Normalized 9', 'Normalized 10', 'Normalized 11', 'Normalized 12', 'Normalized 13', 'Normalized 14', 'Normalized 15', 'Normalized 16', 'Normalized 17', 'Normalized 18', 'Normalized 19', 'Normalized 20', 'Normalized 21', 'Normalized 22', 'Normalized 23', 'Normalized 24', 'Normalized 25', 'Normalized 26', 'Normalized 27', 'Normalized 28', 'Normalized 29', 'Normalized 30', 'Normalized 31', 'Normalized 32', 'Normalized 33', 'Normalized 34', 'Normalized 35', 'Normalized 36', 'Normalized 37', 'Normalized 38', 'Normalized 39', 'Normalized 40', 'Normalized 41', 'Normalized 42', 'Normalized 43', 'Normalized 44', 'Normalized 45', 'Normalized 46', 'Normalized 47', 'Normalized 48', 'Normalized 49', 'Normalized 50', 'Normalized 51']]

kMeansLabels = kMeansClustering(filteredData, 3)[0]
kMeansPPLabels = kMeansClustering(filteredData, 3)[1]
agl = agglomerativeClustering(filteredData, 3)
dbscanClustering(filteredData)
plotSelectedData(filteredData, 'Normalized 0', 'Normalized 1', 'Clustering KMeans', kMeansLabels)
plotSelectedData(filteredData, 'Normalized 0', 'Normalized 1', 'Clustering KMeans++', kMeansPPLabels)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(filteredData)
plt.title('Hierarchical Clustering Dendrogram (Agglomerative Clustering)')

plot_dendrogram(model, truncate_mode='level', p=3)
plt.show()