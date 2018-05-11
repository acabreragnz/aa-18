import sys
from tabulate import tabulate
from scipy.spatial import distance
import matplotlib.pyplot as plt

def print_results(kmeans, clusters):
    # Obtain centroids and number Cluster of each point
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist ()

    cant = 0;
    for i, centroid in enumerate(centroids):

        for j, cluster in enumerate(clusters):
            dist = distance.euclidean(cluster.centroid, centroid)
            if dist == 0 and num_cluster_points.count(i) == len(cluster.points):
                cant = cant + 1

    print(f'\nCantidad de clusters con igual centroide e igual cantidad de puntos: {cant}\n')


def print_results2(J, J_sklearn):

    table = []
    for i in range(len(J)):
        table.append([i+2, J[i], J_sklearn[i]])

    print(tabulate(table, headers=["#Clusters", "Costo", "Costo sklearn"]))

    plt.figure (figsize=(10, 10))
    plt.ylabel ('Función de costo')
    plt.grid (True)

    x = [i+2 for i in range(len(J))]
    plt.plot (x, J, color='r', label='k_means')
    plt.plot (x, J_sklearn, color='g', label='KMeans sklearn')

    plt.legend (loc=0)
    plt.show ()