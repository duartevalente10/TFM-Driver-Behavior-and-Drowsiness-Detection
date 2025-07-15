"""
modeling.clustering
----------------------------

This module provides diferent aproaches to the clustering task:
    - Kmeans
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
# local
from pre_process_1 import (
    read_csv_file,
    min_max_scaler,
    normalize_by_distance,
    normalize_by_duration,
    pca
)


def clusters_info_print(data, y_pred, path=None, show=False):
    """
        Print statistical information of clusters.

        Parameters:
        - data: Input data
        - y_pred: Cluster predictions
        - path: Path to save cluster information
        - show: Whether to display the information
    """
    clusters = pd.Series(y_pred, name='clusters')
    df = pd.merge(data, clusters, left_index=True, right_index=True)
    print('\n --------------------- Cluster Statistical Analysis --------------------- \n')
    print(df.groupby(['clusters']).min().T)
    print(df.groupby(['clusters']).max().T)
    print(df.groupby(['clusters']).mean().T)
    print(df.groupby(['clusters']).std().T)


def visualize_clusters_with_pca(data, target):
    """
        Visualize clusters using PCA for dimensionality reduction.

        Parameters:
        - data: Input data
        - target: Target labels
    """
    model = PCA(n_components=2).fit(data)
    X_pc = model.transform(data)
    print(X_pc.shape)

    plt.figure(figsize=(16,7))
    sns.scatterplot(
        x=X_pc[:, 0],
        y=X_pc[:, 1],
        hue=target, 
        palette=sns.color_palette("hls", 2), 
        data=X_pc, 
        legend="full"
    )
    plt.show()


def analyse_via_pca_components(data, n_components=2):
    """
        Analyze data via PCA components and identify important features.

        Parameters:
        - data: Input data
        - n_components: Number of PCA components

        Returns:
        - most_important_names: List of most important feature names
        - X_pc: Transformed data using PCA
    """
    
    model = PCA(n_components=n_components).fit(data)
    X_pc = model.transform(data)

    # number of components
    n_pcs = model.components_.shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = list(data.columns)

    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # show most important features for fist component
    pc0 = model.components_[0]
    top_n_features = 5
    indexes = sorted(range(len(pc0)), key=lambda sub: pc0[sub])[:-(top_n_features+1):-1]
    most_f_names = [initial_feature_names[i] for i in indexes]
    print('Most important feature for component 0:', most_f_names)

    # build the dataframe
    df = pd.DataFrame(dic.items())
    print(df)

    return most_important_names, X_pc


def analyse_via_decision_tree(data, target, n_top_features=3):
    """
    Analyze data via Decision Tree and identify important features.

    Parameters:
    - data: Input data
    - target: Target labels
    - n_top_features: Number of top features to identify
    """

    dec = DecisionTreeClassifier(max_depth=5)
    dec.fit(data, target)
    feature_importance = dec.feature_importances_
    print('Decision tree - feature importance:', feature_importance)
    print('Decision tree - most important feature:', data.columns[feature_importance.argmax()])
    indexes = sorted(range(len(feature_importance)), key=lambda sub: feature_importance[sub])[:-(n_top_features+1):-1]
    print('Decision tree - Top features:', data.columns[indexes])
    plot_tree(
        dec,
        feature_names=list(data.columns),
        class_names=['0', '1'],
        filled=True,
        fontsize=10
    )
    plt.title("Decision tree")
    plt.show()


def analyse_via_random_forest(data, target, n_top_features=3):
    """
        Analyze data via Random Forest and identify important features.

        Parameters:
        - data: Input data
        - target: Target labels
        - n_top_features: Number of top features to identify
    """
    rfc = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=50)
    rfc.fit(data, target)
    feature_importance = rfc.feature_importances_
    print('Random forest - feature importance:', feature_importance)
    print('Random forest - most important feature:', data.columns[feature_importance.argmax()])
    indexes = sorted(range(len(feature_importance)), key=lambda sub: feature_importance[sub])[:-(n_top_features+1):-1]
    print('Random forest - Top features:', data.columns[indexes])
    plot_tree(
        rfc.estimators_[0],
        feature_names=list(data.columns),
        class_names=['0', '1'],
        filled=True,
        fontsize=8,
        max_depth=4
    )
    plt.title("Random Forest tree")
    plt.show()



if __name__ == "__main__":

    # import data
    df_path = '../datasets/supervised/trips_consensus_kmeans'
    df = read_csv_file(df_path)

    # remove start, and end
    df = df[(df.columns.difference([
        'trip_start', 'trip_end', 'light_mode'
    ], sort=False))]

    # get traget values
    data = df.drop('target', axis=1)
    target = df['target']

    print(len(np.unique(data['speed'])))
    print(len(np.unique(data['n_ha'])))
    print(len(np.unique(data['n_fatigue_1'])))

    # data = normalize_by_duration(data)
    data = min_max_scaler(data)

    # visualize_clusters_with_pca(data, target)

    # path = ''
    # clusters_info(data, target, path=path, show=False)

    # analyze via PCA components
    best_features, X_pc = analyse_via_pca_components(data, n_components=2)

    # scatter plot using most important features
    plt.figure(figsize=(16,7))
    sns.scatterplot(
        x='speed',  # best_features[0], 
        y='n_ha',  # best_features[1], 
        hue=target, 
        palette=sns.color_palette("hls", 2), 
        data=data, 
        legend="full"
    )
    plt.show()

    # analyze via Decision Tree
    analyse_via_decision_tree(data, target, n_top_features=3)

    # analyze via Random Forest
    analyse_via_random_forest(data, target, n_top_features=3)
    