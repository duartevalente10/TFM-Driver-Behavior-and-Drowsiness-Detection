"""
modeling.clustering
----------------------------

This module provides an aproache to the clustering task:
    - ConsensusKmeans
"""

# packages
import biosppy.clustering as bioc
import biosppy.plotting as biop
import biosppy.metrics as biom
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster import hierarchy
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt

class ConsensusKmeans:

    def __init__(self, kmin=None, kmax=None, n_ensemble=100):
        """
            Initialize the ConsensusKmeans object.

            Parameters:
            - kmin: Minimum number of clusters to consider
            - kmax: Maximum number of clusters to consider
            - n_ensemble: Number of ensemble runs
        """
        self.kmin = kmin
        self.kmax = kmax
        self.n_ensemble = n_ensemble

    def ensemble(self, data):
        """
            Perform ensemble clustering using random K-means.

            Parameters:
            - data: Input data for clustering

            Returns:
            - ensemble: Ensemble of clustering results
        """

        N = len(data)

        # determine kmin and kmax if not provided
        if self.kmin is None:
            self.kmin = int(round(np.sqrt(N) / 2.))

        if self.kmax is None:
            self.kmax = int(round(np.sqrt(N)))

        grid = {
            'k': np.random.randint(low=self.kmin, high=self.kmax, size=self.n_ensemble)
        }

        # generate ensemble of clustering results
        ensemble, = bioc.create_ensemble(data.to_numpy(), fcn=bioc.kmeans, grid=grid)
        print(ensemble)
        return ensemble

    def coassoc_matrix(self, ensemble, data_size, path=None, show=False):
        """
            Generate co-association matrix from clustering ensemble.

            Parameters:
            - ensemble: Clustering ensemble
            - data_size: Size of the input data
            - path: Path to save the plot 
            - show: display the plot

            Returns:
            - coassoc: Co-association matrix
        """
        coassoc, = bioc.create_coassoc(ensemble, data_size)
        plt.imshow(coassoc, interpolation='nearest')
        if path is not None:
            plt.savefig(path + '.png')
        if show:
            plt.show()
        return coassoc

    def coassoc_partition(self, coassoc, k, linkage):
        """
            Perform partitioning of co-association matrix.

            Parameters:
            - coassoc: Co-association matrix
            - k: Number of clusters
            - linkage: Linkage criterion for clustering

            Returns:
            - clusters: Partitioned clusters
        """
        clusters, = bioc.coassoc_partition(coassoc, k, linkage)
        return clusters

    def visualize_clusters(self, data, clusters, path=None, show=False):
        """
            Visualize clusters in 3D space.

            Parameters:
            - data: Input data for visualization
            - clusters: Cluster assignments
            - path: Path to save the plot
            - show: Whether to display the plot

            Returns:
            - y_pred: Predicted cluster labels
        """
        # plot clustering visualization
        biop.plot_clustering(data.to_numpy(), clusters, path, show)

        # determine number of clusters
        keys = list(clusters)
        n_rows = len(data)
        y_pred = np.ones((n_rows,), dtype=int)

        for k in keys:
            y_pred[clusters[k]] = k
            # if i == 0:
            #     axis_x = data.iloc[clusters[k], :]

        # Scatter plot for 3D visualization
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(111, projection='3d')
        sc = axis.scatter(
            data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=y_pred
        )
        axis.set_xlabel(data.columns[0], fontsize=10)
        axis.set_ylabel(data.columns[1], fontsize=10)
        axis.set_zlabel(data.columns[2], fontsize=10)
        plt.legend(*sc.legend_elements(), loc=1, title='Clusters')

        if path is not None:
            plt.savefig(path + '.png')
        if show:
            plt.show()
        return y_pred

    def evaluate_clusters(self, data, y_pred, path=None, show=False):
        """
            Evaluate clustering performance using various metrics.

            Parameters:
            - data: Input data for evaluation
            - y_pred: Predicted cluster labels
            - path: Path to save evaluation results
            - show: Whether to display evaluation results
        """
        clusters = np.unique(np.array(y_pred)) 

        # handle case of only one cluster found
        if len(clusters) == 1:
            c_h_score = 'Only one cluster found'
            d_b_score = 'Only one cluster found'
            s_score = 'Only one cluster found'
        else: 
            # calculate clustering evaluation metrics
            c_h_score = calinski_harabasz_score(data, y_pred)
            d_b_score = davies_bouldin_score(data, y_pred)
            s_score = silhouette_score(data, y_pred)
        
        # save evaluation results to file if path provided
        if path:
            with open(path + 'evaluation.txt', 'a+') as f:
                for _, v in enumerate(clusters):
                    n = len(y_pred[y_pred == v])
                    f.write('N instances belonging to cluster {}: {} \n'.format(v, n)) 
                f.write('Calinski score: {} \n'.format(c_h_score))
                f.write('Davies-Bouldin score: {} \n'.format(d_b_score))
                f.write('Silhouette score: {} \n \n'.format(s_score))

        # print evaluation results if show is True
        if show:
            for _, v in enumerate(clusters):
                n = len(y_pred[y_pred == v])
                print('N instances belonging to cluster {}:'.format(v), n) 
            print('Calinski score:', c_h_score)
            print('Davies-Bouldin score:', d_b_score)
            print('Silhouette score:', s_score, '\n')
