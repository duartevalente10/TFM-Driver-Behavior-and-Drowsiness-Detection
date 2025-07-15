"""
preprocess.feature_selection
----------------------------

This module provides diferent aproaches to the feature selection
    (unsupervised) task:

    - Algorithm 1: Relevance-only filter
    - Algorithm 2: Filter based on relevance and redundancy
        - To measure relevance use one of:
            * Mean Absolute Difference (MAD)
            * Arithmetic Mean (AM)
            * Geometric Mean (GM)
            * Arithmetic Mean Geometric Mean Quotient (AMGM)
            * Mean Median (MM)
            * Variance (VAR)
        - To measure redundancy use one of:
            * Absolute cosine (AC)
            * Correlation coefficient (CC)

Other filter based approaches:
    - Term-variance
    - Laplacian Score
    - Spectral methods
"""

# packages
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# local
#from b_construct_dataset import read_csv_file
from .f_normalization import normalize_by_distance, normalize_by_duration

def relevance_filter(df, measure, m):
    """
    Select a subset of features based on the relevance of each feature.
    Sort the features by decreasing order and keep the top m.

    Args:
        df (pandas.DataFrame): Dataset
        measure (str): Relevance measure type, one of:
            - Mean Absolute Difference (MAD)
            - Arithmetic Mean (AM)
            - Geometric Mean (GM)
            - Arithmetic Mean Geometric Mean Quotient (AMGM)
            - Mean Median (MM)
            - Variance (VAR)
        m (int): Top m of features to keep

    Returns:
        pandas.Series: Top m most relevant features
    """

    try:
        assert m <= df.shape[1]
    except AssertionError:
        print("'m' must be <= to the current number of features")
        return None

    if measure == 'MAD':
        relevance = df.mad()
    elif measure == 'AM':
        relevance = df.mean()
    elif measure == 'GM':
        # returns 0 if there are any zeros in column
        # the normalization process can return a lot of zeros for the columns
        # Not a good measure !!!
        relevance = stats.gmean(df.iloc[:, 0])
    elif measure == 'AMGM':
        relevance = np.exp(df).mean() / stats.gmean(np.exp(df))
    elif measure == 'MM':
        relevance = (df.mean() - df.median()).abs()
    elif measure == 'VAR':
        relevance = df.var()

    # sort relevance by decreasing order
    if isinstance(relevance, pd.core.series.Series):
        relevance = relevance.sort_values(ascending=False).index
    else:
        # only for geometric mean
        relevance = np.sort(relevance)[::-1]

    # return top m features
    return relevance[:m]

def relevance_redundancy_filter(df, rev_measure, red_measure, m, ms,target_name):
    """
    Apply relevance filter and then check reduntant features.

    Args:
        df (pandas.DataFrame): Dataset
        rev_measure (str): Relevance measure type, one of:
            - Mean Absolute Difference (MAD)
            - Arithmetic Mean (AM)
            - Geometric Mean (GM)
            - Arithmetic Mean Geometric Mean Quotient (AMGM)
            - Mean Median (MM)
            - Variance (VAR)
        red_measure (str): Redundancy measure type, one of:
            - Correlation coefficient (CC)
            - Absolute cosine (AC)
        m (int): Top m of features to keep
        ms (int): Maximum allowed similarity between pairs of features

    Returns:
        pandas.Series: Top m most relevant features
    """

    try:
        assert m <= df.shape[1]
    except AssertionError:
        print("'m' must be <= to the current number of features")
        return None
    
    #calculate sorted relevance and keep all features
    if rev_measure == 'FR':
        relevance = fisher_ratio(df.drop(columns=[target_name]), df[target_name])
        print("Fisher Ratio Relevance: ",relevance)
    else:
    #calculate sorted relevance and keep all features
        relevance = relevance_filter(df, rev_measure, df.shape[1])
        print(relevance, len(relevance))

    # keep most relevant feature
    feature_keep = np.array([relevance[0]])
    similarities = []

    prev = feature_keep[-1]

    # loop through relevances starting from second
    for i in relevance[1:]:
        # compute similarity (redundancy) of current feature and previous
        if red_measure == 'CC':
            # 0 < CC < 1
            # 0 - min similirity | 1 - max similarity
            s = df[i].corr(df[prev])
        elif red_measure == 'AC':
            # 0 < AC < 1
            # 0 - min similirity (ortogonal) | 1 - max similarity
            s = cosine_similarity(
                df[i].values.reshape(1, -1), df[prev].values.reshape(1, -1)
            )[0, 0]

        #print('Similarity entre', i, 'e', prev, ':', s)

        similarities.append((i, prev, s))

        # TODO: se o metodo de similaridade for o CC também funciona para valores negativos
        if s < ms:
            feature_keep = np.append(feature_keep, i)
            prev = i

        #print(feature_keep, len(feature_keep))

        # we have enough features
        if len(feature_keep) == m:
            break
    #print(similarities)
    
    # Extract feature pairs and similarities
    feature_pairs = [(pair[0], pair[1]) for pair in similarities]
    similarity_values = [pair[2] for pair in similarities]

    # Plot similarities
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(similarity_values)), similarity_values, color='skyblue')
    plt.xlabel('Feature pairs')
    plt.ylabel('Similarity')
    plt.title('Similarities between feature pairs')
    plt.xticks(range(len(similarity_values)), feature_pairs, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return feature_keep

def calculate_relevance(data, th):
    """
    Compute each feature relevance

    Arguments:
    data -- unsupervised normalized data
    th -- threshold of relevance to keep 
    
    Returns:
    Name of the most relevant features that keep the relevance of the indicated th
    """

    # compute the variance and the mean - median
    variance = data.var()
    mean_median = (data.mean() - data.median()).abs()

    # combine the computed values into a dataframe
    relevance = pd.DataFrame({'variance': variance, 'mean_median': mean_median})
    
    # sort relevance values in descending order
    sorted_relevance = relevance.sort_values(by='variance', ascending=False)

    print("Relevance v1: ",sorted_relevance)
    
    # create dataframe of sorted relevance values
    sorted_relevance_data = pd.DataFrame({'relevance': sorted_relevance['variance'].values, 'rank': range(1, len(sorted_relevance) + 1)})
    
    # plot the relevance values
    plt.figure()
    plt.bar(sorted_relevance_data['rank'], sorted_relevance_data['relevance'])
    plt.xlabel('# Features (m)')
    plt.title('Relevance by (Variance and (Mean - Median))')
    plt.show()
    
    # compute cumulative sum and total sum of relevance values
    cumulative_sum = sorted_relevance['variance'].cumsum()
    total_sum = sorted_relevance['variance'].sum()
    
    # compute cumulative proportion of relevance values
    cumulative_proportion = cumulative_sum / total_sum
    
    
    # find index of first relevance value that exceeds the threshold
    m_value = next((index for index, value in enumerate(cumulative_proportion) if value >= th), None)

    # get features that exceed the threshold
    features = sorted_relevance.index[:m_value + 1]
    
    return features

def fisher_ratio(features, labels, max_features=None):
    """
    Compute the Fisher ratio of relevance of each feature with the class label vector
    and return features ordered by their Fisher ratio scores.

    Args:
        features (pandas.DataFrame): DataFrame containing feature vectors.
        labels (pandas.Series): Class label vector.

    Returns:
        list: Features ordered by their Fisher ratio scores in descending order.
    """
    fisher_scores = {}
    for feature_name in features.columns:
        feature = features[feature_name]
        unique_labels = labels.unique()
        class_means = {label: feature[labels == label].mean() for label in unique_labels}
        overall_mean = feature.mean()
        
        between_class_variance = sum([len(feature[labels == label]) * (class_means[label] - overall_mean)**2 for label in unique_labels])
        between_class_variance /= (len(unique_labels) - 1)
        
        within_class_variance = sum([((feature[labels == label] - class_means[label])**2).sum() for label in unique_labels])
        within_class_variance /= (len(feature) - len(unique_labels))
        
        fisher_scores[feature_name] = between_class_variance / within_class_variance
    
    # Sort features by Fisher ratio scores in descending order
    sorted_features = sorted(fisher_scores, key=fisher_scores.get, reverse=True)

    # Limit the number of features to plot
    if max_features is not None:
        sorted_features = sorted_features[:max_features]
    
    # Plot Fisher ratio scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_features)), [fisher_scores[feature] for feature in sorted_features], color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Fisher Ratio Score')
    plt.title('Fisher Ratio Scores of Features')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return sorted_features


if __name__ == "__main__":

    # Load your dataset
    #trips = pd.read_csv('datasets/categorical_data/trips_label_encoding.csv')
    #trips = pd.read_csv('../datasets/normalization/trips_norm_distance.csv')
    #trips = pd.read_csv('../datasets/supervised/trips_kmeans_no_reduction_distance.csv')
    #trips = pd.read_csv('../datasets/missing_values/trips_mv_all.csv')
    trips = pd.read_csv('../datasets/supervised/trips_kmeans_2_stage_RRFS.csv')

    # trips = trips[(trips.columns.difference([
    #     'trip_start', 'trip_end', 'light_mode', 'zero_speed_time', 'n_zero_speed', 'n_ignition_on',
    #     'n_ignition_off', 'n_high_beam', 'n_low_beam', 'n_wipers', 'n_signal_right', 'n_signal_left'
    # ], sort=False))]

    # norm_distance = normalize_by_distance(trips)

    #print('Trips feature size before:', trips.shape[1])

    # ------------- Relevance Filters ------------- #
    # print('---------------------------------------------')
    # print('------------- Relevance Filters -------------')
    # print('---------------------------------------------')
    # print('------ Mean Absolute Difference ------')
    # features = relevance_filter(trips, 'MAD', 0.99)
    # print(features, len(features), '\n')
    # print('------ Arithmetic Mean ------')
    # features1 = relevance_filter(trips, 'AM', 0.99)
    # print(features1, len(features1), '\n')
    #print('------ Geometric Mean ------')
    #features2 = relevance_filter(trips, 'GM', 30)
    #print(features2, len(features2), '\n')
    #print('------ Arithmetic Mean Geometric Mean Quotient ------')
    #features3 = relevance_filter(trips, 'AMGM', 30)
    #print(features3, len(features3), '\n')
    # print('------ Mean Median ------')
    # features4 = relevance_filter(trips, 'MM', 0.99)
    # print(features4, len(features4), '\n')
    # print('------ Variance ------')
    # features5 = relevance_filter(trips, 'VAR', 0.99)
    # print(features5, len(features5), '\n')

    # ------------- Relevance Redundancy Filters ------------- #
    # print('--------------------------------------------------------')
    # print('------------- Relevance Redundancy Filters -------------')
    # print('--------------------------------------------------------')
    # features6 = relevance_redundancy_filter(trips, 'AMGM', 'AC', 30, 0.6)
    # print(features6, len(features6))

    print('--------------------------------------------------------')
    print('--------------------- Relevance ------------------------')
    print('--------------------------------------------------------')

    features = fisher_ratio(trips.drop(columns=['target']), trips['target'], 15)
    print("Fisher Ratio Features: ", features[:5])


    # features = relevance_redundancy_filter(trips, 'FR', 'AC', 6, 0.5)
    # print("Features selecionadas RRFS com Fisher Ratio: ", features)

    #X_train = norm_distance[features1]
    #print(X_train, type(X_train))

    # ------------- TEST KMEANS ------------- #
    #from sklearn.cluster import KMeans
    #train_set = X_train[:7293]
    #test_set = X_train[7293:]
    #print('Train size:', train_set.shape)
    #print('Test size:', test_set.shape)

    #kmeans = KMeans(n_clusters=3, random_state=0).fit(train_set)
    #print('Labels:', kmeans.labels_, len(kmeans.labels_))
    #predicted = kmeans.predict(test_set)
    #print('Prediction:', predicted, len(predicted))
    #print('Clusters:', kmeans.cluster_centers_)

    #X_train['cluster'] = np.concatenate((kmeans.labels_, predicted))
    #for c1 in X_train.columns[:-1]:
    #    for c2 in X_train.columns[:-1]:
    #        sns.scatterplot(
    #            data=X_train, x=c1,
    #            y=c2,
    #            hue="cluster",
    #            style="cluster"
    #        )
    #        plt.show()

    #pd.options.display.float_format = '{:,.3f}'.format
    #plt.figure(figsize=(20, 20))
    #annot_kws={'size': 5}
    #annot=True
    #correlation = X_train.corr()
    #sns.heatmap(
    #    correlation, linewidths=.3, vmax=1, vmin=-1, center=0, cmap='vlag'
    #)
    #correlation = correlation.unstack()
    #correlation = correlation[abs(correlation) >= 0.7]
    #plt.show()
    #print(correlation.to_string())
