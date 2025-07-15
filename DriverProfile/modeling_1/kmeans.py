# packages
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer,
    random_center_initializer
)
from pyclustering.cluster.encoder import cluster_encoder


class KmeansClustering:
    """
        KmeansClustering class for performing KMeans clustering with customizable metrics and initialization.
    """

    def __init__(self, n_clusters, init, random_state=None, metric='EUCLIDEAN', data=None):
        """
            Initialize the KmeansClustering object.

            Parameters:
            - n_clusters: Number of clusters
            - init: Initialization method
            - random_state: Seed for random number generator
            - metric: Distance metric to use
            - data: Input data
        """

        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.k_means = self.__init_kmeans(data)

    def __init_kmeans(self, data):  
        """
            Initialize the KMeans algorithm with the chosen initializer and metric.

            Parameters:
            - data: Input data

            Returns:
            - k_means: Initialized KMeans object
        """ 
        initial_centers = self.__choose_initializer(data)
        k_means = kmeans(
            data, initial_centers=initial_centers, metric=self.__choose_metric()
        )
        return k_means

    def __choose_initializer(self, data):
        """
            Choose the initializer for KMeans.

            Parameters:
            - data: Input data

            Returns:
            - initial_centers: Initial cluster centers
        """
        if self.init == 'random':
            initial_centers = random_center_initializer(
                data.values, self.n_clusters, random_state=self.random_state
            ).initialize()
        else:
            initial_centers = kmeans_plusplus_initializer(
                data.values, self.n_clusters, random_state=self.random_state
            ).initialize()
        return initial_centers

    def __choose_metric(self):
        """
            Choose the distance metric for KMeans.

            Returns:
            - metric: Chosen distance metric
        """
        if self.metric == 'EUCLIDEAN':
            metric = distance_metric(type_metric.EUCLIDEAN)
        elif self.metric == 'EUCLIDEAN_SQUARE':
            metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)
        elif self.metric == 'MANHATTAN':
            metric = distance_metric(type_metric.MANHATTAN)
        elif self.metric == 'CHEBYSHEV':
            metric = distance_metric(type_metric.CHEBYSHEV)
        elif self.metric == 'CANBERRA':
            metric = distance_metric(type_metric.CANBERRA)
        elif self.metric == 'CHI_SQUARE':
            metric = distance_metric(type_metric.CHI_SQUARE)
        else:
            # by default use euclidean distance
            metric = distance_metric(type_metric.EUCLIDEAN)
        return metric

    def fit_predict(self, data):
        """
            Fit the KMeans algorithm and predict cluster labels.

            Parameters:
            - data: Input data

            Returns:
            - y_pred: Predicted cluster labels
        """
        self.k_means.process()
        # y_pred = self.k_means.predict(data)
        clusters = self.k_means.get_clusters()
        encoding = self.k_means.get_cluster_encoding()
        encoder = cluster_encoder(encoding, clusters, data.values)
        y_pred = encoder.set_encoding(0).get_clusters()  
        return y_pred

    def visualize_clusters(self, data, y_pred, show=False, path=None):
        """
            Visualize the clusters in 2D or 3D.

            Parameters:
            - data: Input data
            - y_pred: Predicted cluster labels
            - show: Whether to display the plot
            - path: Path to save the plot
        """
        dim = True
        # show clusters
        mpl.style.use('default')
        if data.shape[1] > 2:
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111, projection='3d')
            sc = axis.scatter(
                data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=y_pred
            )
            axis.set_xlabel(data.columns[0], fontsize=10)
            axis.set_ylabel(data.columns[1], fontsize=10)
            axis.set_zlabel(data.columns[2], fontsize=10)
            plt.legend(*sc.legend_elements(), loc=1, title='Clusters')
        elif data.shape[1] > 1:
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred)
            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            plt.title('Clusters')
        else:
            print('Only one dimension found!!!')
            dim = False
        if path and dim:
            plt.savefig(path + '.png')
        if show and dim:
            plt.show()

    def evaluate_clusters(self, data, y_pred, show=False, path=None):
        """
            Evaluate the clusters using different metrics.

            Parameters:
            - data: Input data
            - y_pred: Predicted cluster labels
            - show: Whether to display the evaluation
            - path: Path to save the evaluation
        """
        y_pred = np.array(y_pred)
        clusters = np.unique(y_pred) 
        if len(np.unique(y_pred)) == 1:
            c_h_score = 'Only one cluster found'
            d_b_score = 'Only one cluster found'
            s_score = 'Only one cluster found'
        else: 
            c_h_score = calinski_harabasz_score(data, y_pred)
            d_b_score = davies_bouldin_score(data, y_pred)
            s_score = silhouette_score(data, y_pred)

        if path:
            with open(path + '/evaluation.txt', 'a+') as f:
                f.write('------------ {} DISTANCE ------------ \n'.format(self.metric))
                for _, v in enumerate(clusters):
                    n = len(y_pred[y_pred == v])
                    f.write('N instances belonging to cluster {}: {} \n'.format(v, n)) 
                f.write('Calinski score: {} \n'.format(c_h_score))
                f.write('Davies-Bouldin score: {} \n'.format(d_b_score))
                f.write('Silhouette score: {} \n \n'.format(s_score))
        if show:
            print('------------ {} DISTANCE ------------'.format(self.metric))
            for _, v in enumerate(clusters):
                n = len(y_pred[y_pred == v])
                print('N instances belonging to cluster {}:'.format(v), n) 
            print('Calinski score:', c_h_score)
            print('Davies-Bouldin score:', d_b_score)
            print('Silhouette score:', s_score, '\n')
    
    def __repr__(self):
        txt = "<KmeansClustering(n_clusters={}, metric={})>"
        return txt.format(self.n_clusters, self.metric)