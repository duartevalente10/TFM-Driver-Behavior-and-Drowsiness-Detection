# packages
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class GaussianMixtureClustering:
    """
    Representation of a Gaussian mixture model probability distribution for
    clustering analysis
    """

    def __init__(self, n_clusters, covariance_type='full', init_params='kmeans', random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.gm = GaussianMixture(
            n_components=self.n_clusters, 
            random_state=self.random_state,
            covariance_type=self.covariance_type,
            init_params=init_params
        )

    def fit_predict(self, data):
        return self.gm.fit_predict(data)

    def visualize_clusters(self, data, y_pred, path=None, show=False):
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

    def evaluate_clusters(self, data, y_pred, path=None, show=False):
        clusters = np.unique(y_pred) 
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
                f.write('------------ {} COVARIANCE ------------ \n'.format(self.covariance_type))
                for _, v in enumerate(clusters):
                    n = len(y_pred[y_pred == v])
                    f.write('N instances belonging to cluster {}: {} \n'.format(v, n)) 
                f.write('Calinski score: {} \n'.format(c_h_score))
                f.write('Davies-Bouldin score: {} \n'.format(d_b_score))
                f.write('Silhouette score: {} \n \n'.format(s_score))
        if show:
            print('------------ {} COVARIANCE ------------'.format(self.covariance_type))
            for _, v in enumerate(clusters):
                n = len(y_pred[y_pred == v])
                print('N instances belonging to cluster {}:'.format(v), n) 
            print('Calinski score:', c_h_score)
            print('Davies-Bouldin score:', d_b_score)
            print('Silhouette score:', s_score, '\n')

    def __repr__(self):
        txt = "<GaussianMixtureClustering(n_components={})>"
        return txt.format(self.n_clusters)
    