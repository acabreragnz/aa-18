import random
import numpy as np
import logging

import sys
from scipy.spatial import distance
from lab4.ej1.src.kmeans.Point import Point
from lab4.ej1.src.kmeans.Cluster import Cluster


# Se inicializan al azar los K centroides u1, u2, ... uk
#
# Sean m los ejemplos de entrenamiento:  {x1, x2, ... ,xm}
# Repetir hasta que el algoritmo converja:
# 	Para cada i entre 1 y m (para cada ejemplo de entrenamiento):
#       (Lo asignamos al cluster cuyo centroide es más cercano)
#       c(i) := argminj ||xi - uj||^2
# 	Para cada k entre 1 y k (para cada cluster):
#       ui = ∑ 1 {ci = j} xi / ∑ 1 {ci = j}  ¥ i = 1..m
#Dónde:
#    1 {ci = j} = 1 si ci = j, cero en otro caso

def dataset_to_list_points(dataset):
    """
    dataset es el texto transformado en características utilizando CountVectorizer de scikit­learn.
    :param dir_dataset:
    """
    points = list()
    for point in dataset:
            points.append(Point(point))
    return points

def get_nearest_cluster(clusters, point):
    """
    Calculate the nearest cluster
    :param clusters: old clusters
    :param point: point to assign cluster
    :return: index of list cluster
    """
    dist = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        dist[i] = distance.euclidean(point.coordinates, c.centroid)
    return np.argmin(dist)


def print_clusters_status(it_counter, clusters):
    logging.info('\nITERATION %d' % it_counter)
    for i, c in enumerate(clusters):
        logging.info('\tCentroid Cluster %d: %s' % (i + 1, str(c.centroid)))


def print_results(clusters):
    logging.info('\n\nFINAL RESULT:')
    for i, c in enumerate(clusters):
        logging.info('\tCluster %d' % (i + 1))
        logging.info('\t\tNumber Points in Cluster %d' % len(c.points))
        # logging.info('\t\tCentroid: %s' % str(c.centroid))


def k_means(dataset, num_clusters, iterations, optimize=False):
    points = dataset_to_list_points (dataset)

    # J(c,µ) = ∑ || x^(i) - µc^(i)||^2
    # Se inicializan varias veces los centroides y se toma el resultado con menor valor en la funcion de costos
    # para evitar minimos locales

    clusters_cost_min = None
    cost_min = sys.float_info.max

    n = 1
    if optimize:
        n = 5

    for i in range (1):
        # INICIALIZACIÓN: Selección aleatoria de N puntos y creación de los Clusters
        initial = random.sample (points, num_clusters)
        clusters = [Cluster ([p]) for p in initial]

        # Inicializamos una lista para el paso de asignación de objetos
        new_points_cluster = [[] for i in range (num_clusters)]

        converge = False
        it_counter = 0
        while (not converge) and (it_counter < iterations):
            # ASIGNACION
            for p in points:
                i_cluster = get_nearest_cluster (clusters, p)
                new_points_cluster[i_cluster].append (p)

            # ACTUALIZACIÓN
            for i, c in enumerate (clusters):
                c.update_cluster (new_points_cluster[i])

            # ¿CONVERGE?
            converge = [c.converge for c in clusters].count (False) == 0

            # Incrementamos el contador
            it_counter += 1
            new_points_cluster = [[] for i in range (num_clusters)]

        cost = 0
        for c in clusters:
            cost = cost + c.cost_function ()
        if cost < cost_min:
            cost_min = cost
            clusters_cost_min = clusters

    return clusters_cost_min