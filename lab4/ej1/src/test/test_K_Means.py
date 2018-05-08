from unittest import TestCase

import sys
from kmeans.k_means import k_means
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn.cluster import KMeans
from tabulate import tabulate
from scipy.spatial import distance

import logging

class TestKMeans(TestCase):

    def test(self):

        logging.basicConfig(filename='./logs/test_kmeans.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        dataset = load_files ("../../../datasets",categories=['Health-Tweets'])
        # list of text documents
        text = dataset.data
        # create the transform
        vectorizer = CountVectorizer (encoding='latin-1')
        # tokenize and build vocab
        vectorizer.fit (text)
        vector = vectorizer.fit_transform (text)

        # summarize encoded vector
        print (vector.shape)
        print (type (vector))
        # print (vector.toarray ())

        max_iterations =300
        for n_clusters in range(2,16):
            logging.info (f'Cluseters: {n_clusters}')
            clusters = k_means(vector, n_clusters, max_iterations)
            logging.info (
                '-------------------------------------------')
            t0 = time ()
            kmeans = KMeans (n_clusters=n_clusters, max_iter=max_iterations, init='random')
            # Calculate Kmeans
            kmeans.fit (vector)

            # Print final result
            print_results (kmeans, clusters)

        logging.info ('------------------------------------------------------------------------------------------------')

def print_results(kmeans, clusters):
    # Obtain centroids and number Cluster of each point
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist ()
    clusters_copy = clusters[:]
    print ('\n\nFINAL RESULT:')
    table = []
    for i, centroid in enumerate(centroids):
        dist_min = sys.float_info.max
        cluster_prox = None
        for j, cluster in enumerate(clusters_copy):
            dist = distance.euclidean(cluster.centroid, centroid)
            if dist <= dist_min:
                dist_min = dist
                cluster_prox = cluster
        table.append ([i + 1, num_cluster_points.count (i), clusters.index(cluster_prox) + 1, len (cluster_prox.points),
                       distance.euclidean (cluster_prox.centroid, centroid)])
        clusters_copy.remove(cluster_prox)

    print (tabulate (table, headers=["Cluster", "Number Points in Cluster sklearn", "Cluster", "Number Points in Cluster", "Distancia centroides"]))
