import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns
from g_feature_reduction import svd 
from g_feature_selection import relevance_redundancy_filter 

# read dataset
df = pd.read_csv('../datasets_1/supervised/trips_kmeans_2_stage_RRFS_SVD.csv')

X = df.drop('target', axis=1) 

features = relevance_redundancy_filter(X, 'MM', 'AC', 6, 0.5)
print("Features selecionadas RRFS: ", features)
X = X[features]

# PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)

#SVD
# n_components = X.shape[1] - 1
# _, _, best_n_comp = svd(X, None, n_components, 0.99, debug=False)
# svd_result, _, _ = svd(X, None, best_n_comp, 0.99, debug=False)


df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

# t-SNE transformation with 3D output
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300) 
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# Add t-SNE results to the dataframe
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]
#df['tsne-3d-three'] = tsne_results[:, 2]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="target",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)
 
plt.show()

# # Define custom colors for each target
# custom_colors = {
#     0: '#ecaca8',    
#     1: '#e5c78d',  
#     2: '#daecab'     
# }

# # Map the target values to colors
# df['color'] = df['target'].map(custom_colors)

# # Create a 3D plot
# fig = plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     xs=df["tsne-3d-one"], 
#     ys=df["tsne-3d-two"], 
#     zs=df["tsne-3d-three"], 
#     c=df['color'],
#     alpha=0.2
# )

# ax.set_xlabel('tsne-3d-one')
# ax.set_ylabel('tsne-3d-two')
# ax.set_zlabel('tsne-3d-three')

# # Create a legend manually
# handles = [
#     plt.Line2D([0], [0], marker='o', color='w', label='Target 0', markerfacecolor='#ecaca8', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='Target 1', markerfacecolor='#e5c78d', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='Target 2', markerfacecolor='#daecab', markersize=10)
# ]
# ax.legend(handles=handles, loc='best')

# plt.show()