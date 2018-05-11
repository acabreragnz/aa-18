from unittest import TestCase

from kmeans.k_means import k_means
from kmeans.kmeans_helper import print_results, print_results2
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd

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

        df = pd.DataFrame (data=vector.toarray ())
        points = df.as_matrix ().transpose().tolist()
        print (df.as_matrix().shape)

        J = []
        J_sklearn = []
        max_iterations =300
        for n_clusters in range(1000,1000*4,1000):
            logging.info (f'Cluseters: {n_clusters}')

            clusters = k_means(points, n_clusters, max_iterations)
            logging.info (
                '-------------------------------------------')

            kmeans = KMeans (n_clusters=n_clusters, max_iter=max_iterations, init='random')
            # Calculate Kmeans
            kmeans.fit (points)

            # Print final result
            # print_results (kmeans, clusters)

            cost = 0
            for c in clusters:
                cost = cost + c.cost_function ()
            cost_sklearn = kmeans.inertia_
            J.append(cost)
            J_sklearn.append(cost_sklearn)

        print_results2(J, J_sklearn)
        logging.info ('------------------------------------------------------------------------------------------------')


    def test3(self):
        dataset = load_files ("../../../datasets",categories=['Health-Tweets'])
        # list of text documents
        text = dataset.data

        # create the transform
        vectorizer = CountVectorizer (encoding='latin-1')
        # tokenize and build vocab
        vectorizer.fit (text)
        vector = vectorizer.fit_transform (text)

        data = pd.DataFrame (data=vector.toarray())

        print(data.as_matrix().transpose().shape)
        print (data.as_matrix ().transpose ().tolist())
