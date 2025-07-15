# packages
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class DBSCANClustering:
    """
    Density-based spatial clustering of applications with noise
    """

    def __init__(self, eps, min_pts, metric):
        """
        Args:
            eps (float): The distance that specifies the neighborhoods.
                Two points are considered to be neighbors if the distance between
                them are less than or equal to eps.
            min_pts (int): Minimum number of data points to define a cluster.
                How to choose min_ps:
                    * The larger the data set, the larger the value of MinPts should be
                    * If the data set is noisier, choose a larger value of MinPts
                    * Generally, MinPts should be greater than or equal to the
                        dimensionality of the data set
                    * For 2-dimensional data, use DBSCAN's default value of
                        MinPts=4 (Ester et al., 1996)
                    * If your data has more than 2 dimensions, choose MinPts=2*dim
                        where dim=the dimensions of your data set (Sander et al., 1998)
        """
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        self.db = DBSCAN(eps=self.eps, min_samples=self.min_pts, metric=self.metric)

    def fit_predict(self, data):
        return self.db.fit_predict(data)

    def visualize_clusters(self, data, y_pred, path=None, show=False):
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
        else:
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred)
            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            plt.title('Clusters')
        if path:
            plt.savefig(path + '.png')
        if show:
            plt.show()

    def evaluate_clusters(self, data, y_pred, path=None, show=False):
        clusters = np.unique(self.db.labels_) 

        if len(clusters) == 1:
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
    