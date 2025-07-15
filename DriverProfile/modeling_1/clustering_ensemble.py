"""
modeling.clustering
----------------------------

This module provides an implemetation of the clustering task:
    - ConsensusKmeans
"""

import sys
sys.path.append("..")

# packages
import pandas as pd
# local

from modeling_1.consensusKmeans import ConsensusKmeans
from pre_process_1 import normalize_by_distance, pca, svd, relevance_filter, relevance_redundancy_filter, calculate_relevance, normalize_by_duration

if __name__ == "__main__":

    # import dataset
    df = pd.read_csv('../datasets/missing_values/trips_mv_all.csv')

    # remove variables that dont relate to the objective of this thesis
    df = df[(df.columns.difference([
        'trip_start', 'trip_end', 'light_mode', 'zero_speed_time', 'n_zero_speed', 'n_ignition_on', 
        'n_ignition_off', 'n_high_beam', 'n_low_beam', 'n_wipers', 'n_signal_right', 'n_signal_left'
    ], sort=False))]

    print('Dataset shape:', df.shape)

    # normalize
    norm_distance = normalize_by_distance(df)
    print(norm_distance.columns.tolist())
    norm_duration = normalize_by_duration(df)

    features2 = calculate_relevance(df, 0.80)
    print("Features selecionadas FS MM: ", features2)
    print("Numero de features selecionadas FS MM: ", len(features2))
    X_train_FS = df[features2]
    
    # Feature Selection

    # PCA
    X_train_pca, _ = pca(X_train_FS, None, 0.99, False)

    ## SVD
    # n_components = features.shape[1] - 1
    # _, _, best_n_comp = svd(features, None, n_components, 0.99, debug=False)
    # X_train_svd, _, _ = svd(features, None, best_n_comp, 0.99, debug=False)

    # Feature Reduction

    ## FS (mean-median)
    # features = relevance_filter(norm_distance, 'MAD', 2)
    # X_train_FS = norm_distance[features]
    
    ## RRFS 
    # features2 = relevance_redundancy_filter(norm_distance, 'MM', 'AC', 6, 0.5)
    # print("Features selecionadas RRFS: ", features2)
    # X_train_FS = norm_distance[features2]


    # consensus kmeans params
    kmin = 10 
    kmax = 30
    n_ensemble = 250
    linkages = ['average','complete', 'single', 'weighted']
    l = 'average'
    norm = 'distance_norm'
    
    # ensemble model
    ck = ConsensusKmeans(kmin, kmax, n_ensemble)
    print("Ensemble")
    clusters = ck.ensemble(X_train_pca)
    
    # compute Co Association matrix
    print("Coassoc Matrix")
    path='./images/unsupervised/consensus_kmeans/{}/pca_coassoc_'.format(norm) + l
    coassoc = ck.coassoc_matrix(clusters, len(X_train_pca), path=None, show=True)

    # compute Co Association Partition
    print("Coassoc Partition")
    k = 2  # number of clusters
    clusters = ck.coassoc_partition(coassoc, k, l)

    # visualize computed clusters
    path='./images/unsupervised/consensus_kmeans/{}/no_red_'.format(norm)
    y_pred = ck.visualize_clusters(X_train_pca, clusters, path=None, show=True) 

    # evaluate results
    ck.evaluate_clusters(X_train_pca, y_pred, path=path, show=True)


