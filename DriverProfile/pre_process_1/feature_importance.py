# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# dataset
#trips = pd.read_csv('../datasets/supervised/trips_kmeans_no_reduction_distance.csv')
trips = pd.read_csv('../datasets/supervised/trips_kmeans_no_reduction_duration.csv')

# separar os dados das labels
X = trips.drop(columns=['target']) 
y = trips['target']

# print('\n ----------------- Random Forest ----------------- \n')

# # iniciar o random forest 
# rf = RandomForestClassifier()

# # fit do modelo nos dados
# rf.fit(X, y)

# # obter a importancia de cada feature
# importances = rf.feature_importances_

# # dataframe mara mostrar a importancia das features
# feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])

# # ordenar a importancia
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # n features mais importantes
# n = 10
# print("Top", n, "Most Important Features:")
# print(feature_importance_df.head(n))

# # plot das importancias das features
# feature_importance_df.head(n).plot(kind='barh')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Top ' + str(n) + ' Most Important Features')
# plt.show()

# print('\n ----------------- Decision Tree ----------------- \n')

# # iniciar o Decision Tree classifier
# dt = DecisionTreeClassifier()

# # fit do modelo nos dados
# dt.fit(X, y)

# # obter a importancia de cada feature
# importances = dt.feature_importances_

# # dataframe mara mostrar a importancia das features
# feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])

# # ordenar a importancia
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # n features mais importantes
# n = 10  
# print("Top", n, "Most Important Features:")
# print(feature_importance_df.head(n))

# # plot das importancias das features
# feature_importance_df.head(n).plot(kind='barh')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Top ' + str(n) + ' Most Important Features')
# plt.show()

# print('\n ----------------- XGBoost ----------------- \n')

# # iniciar o XGBoost classifier
# xgb_model = xgb.XGBClassifier()

# # fit do modelo nos dados
# xgb_model.fit(X, y)

# # obter a importancia de cada feature
# importances = xgb_model.feature_importances_

# # dataframe mara mostrar a importancia das features
# feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])

# # ordenar a importancia
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # n features mais importantes
# n = 10 
# print("Top", n, "Most Important Features:")
# print(feature_importance_df.head(n))

# # plot das importancias das features
# feature_importance_df.head(n).plot(kind='barh')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Top ' + str(n) + ' Most Important Features')
# plt.show()

print('\n ----------------- SVM ----------------- \n')

# iniciar o SVM
svm_model = SVC(kernel='linear')

# fit do modelo nos dados
svm_model.fit(X, y)

# obter os coeficientes de cada feature
coefficients = svm_model.coef_[0]

# obter os valores absolutos de cada coeficiente
importances = np.abs(coefficients)

# dataframe mara mostrar a importancia das features
feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])

# ordenar a importancia
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# n features mais importantes
n = 10  
print("Top", n, "Most Important Features:")
print(feature_importance_df.head(n))

# plot das importancias das features
feature_importance_df.head(n).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top ' + str(n) + ' Most Important Features')
plt.show()




