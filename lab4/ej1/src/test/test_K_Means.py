from unittest import TestCase

from kmeans.k_means import k_means
from kmeans.kmeans_helper import print_results, print_results2
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import glob

class TestKMeans(TestCase):

    def test(self):

        path = '../../../datasets/Health-Tweets/*.txt'
        filenames = glob.glob (path)

        text = []
        for fname in filenames:
            with open (fname,encoding='latin-1') as infile:
                for line in infile:
                    text.append(line)

        vectorizer = CountVectorizer (encoding='latin-1')
        vector = vectorizer.fit_transform (text)

        print(vector.shape)

        df = pd.DataFrame (data=vector.toarray ())
        points = df.as_matrix ().tolist()
        print (df.as_matrix().shape)

        J = []
        J_sklearn = []
        max_iterations =10
        for n_clusters in range(100,101):

            clusters = k_means(points, n_clusters, max_iterations)

            kmeans = KMeans (n_clusters=n_clusters, max_iter=max_iterations, init='random')
            # Calculate Kmeans
            kmeans.fit (points)

            # Print final result
            print_results (kmeans, clusters)

            cost = 0
            for c in clusters:
                cost = cost + c.cost_function ()
            cost_sklearn = kmeans.inertia_
            J.append(cost)
            J_sklearn.append(cost_sklearn)

        print_results2(J, J_sklearn)



    def test3(self):

        path = '../../../datasets/Health-Tweets/*.txt'
        filenames = glob.glob (path)

        text = []
        for fname in filenames:
            with open (fname,encoding='latin-1') as infile:
                for line in infile:
                    text.append(line)

        vectorizer = CountVectorizer (encoding='latin-1')
        vectors = vectorizer.fit_transform (text)

        print(vectors.shape)

