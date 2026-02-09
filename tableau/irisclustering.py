from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

#
# Prepare the dataset
iris = datasets.load_iris()
print(iris.DESCR)

features, targets = iris.data, iris.target
feature_names = np.char.title(np.char.replace(iris.feature_names, ' (cm)', ''))
target_names = np.char.title(iris.target_names)

dfo = pd.DataFrame(features, columns=feature_names)
print(dfo.describe())

bins = 50
datasets = ['Original', 'Standardized', 'Log10', 'PCA', 'PetalOnly']

#
# Store clustering results
res_cols = ['init', 'n_init', 'max_iter', 'tol', 'dataset', 'missed']
df_res = pd.DataFrame([], columns=res_cols)

#
# Visualize the input data
print(f'\n{datasets[0]}')
dfo.hist(bins=bins)
plt.show()

#
# Standardize the features
ss = StandardScaler()
x_scaled = ss.fit_transform(dfo)
dfs = pd.DataFrame(x_scaled, columns=feature_names)
print(f'\n{datasets[1]}')
dfs.hist(bins=bins)
plt.show()

#
# Transform the features by applying log10 function
dfl = dfo.apply(np.log10)
print(f'\n{datasets[2]}')
dfl.hist(bins=bins)
plt.show()

#
# Reduce dimensions using PCA
pca = PCA(n_components=2)
x_transformed = pca.fit_transform(dfs)

#
# Principal components after reduced
dfp = pd.DataFrame(x_transformed, columns=['Component-1', 'Component-2'])
print(f'\n{datasets[3]}')
dfp.hist(bins=bins)
plt.show()

#
# Dataset containing only petal features
dfpetal = dfo.drop(['Sepal Length', 'Sepal Width'], axis=1)
print(f'\n{datasets[4]}')
dfpetal.hist(bins=bins)
plt.show()

#
# Model parameters
initparam, n_initparam, max_iterparam, tolparam = 'k-means++', 'auto', 300, 0.001

#
# Instantiate a KMeans estimator
km = KMeans(n_clusters=3, random_state=0,
            init=initparam,
            n_init=n_initparam,
            max_iter=max_iterparam,
            tol=tolparam)

#
#-- Fit the model and get the predictions

# Original dataset
km.fit(dfo)
y_pred = km.predict(dfo)

y_predo = np.array(list(map(lambda e: 0 if e == 1 else 1 if e == 0 else 2, y_pred)))
missed = len(y_predo[targets != y_predo])
df_res = pd.concat([df_res,
                   pd.DataFrame([[initparam, n_initparam, max_iterparam, tolparam, datasets[0], missed]],
                                columns=res_cols)])
# Dataset standardized
km.fit(dfs)
y_pred = km.predict(dfs)

y_preds = np.array(list(map(lambda e: 0 if e == 1 else 1 if e == 0 else 2, y_pred)))
missed = len(y_preds[targets != y_preds])
df_res = pd.concat([df_res,
                   pd.DataFrame([[initparam, n_initparam, max_iterparam, tolparam, datasets[1], missed]],
                                columns=res_cols)])

# Dataset transformed with log10 function
km.fit(dfl)
y_pred = km.predict(dfl)

y_predl = np.array(list(map(lambda e: 0 if e == 1 else 1 if e == 0 else 2, y_pred)))
missed = len(y_predl[targets != y_predl])
df_res = pd.concat([df_res,
                   pd.DataFrame([[initparam, n_initparam, max_iterparam, tolparam, datasets[2], missed]],
                                columns=res_cols)])

# Dataset with principal components
km.fit(dfp)
y_pred = km.predict(dfp)

y_predp = np.array(list(map(lambda e: 0 if e == 1 else 1 if e == 0 else 2, y_pred)))
missed = len(y_predp[targets != y_predp])
df_res = pd.concat([df_res,
                   pd.DataFrame([[initparam, n_initparam, max_iterparam, tolparam, datasets[3], missed]],
                                columns=res_cols)])

# Dataset with only petal features
km.fit(dfpetal)
y_pred = km.predict(dfpetal)

y_predpetal = np.array(list(map(lambda e: 0 if e == 1 else 1 if e == 0 else 2, y_pred)))
missed = len(y_predpetal[targets != y_predpetal])
df_res = pd.concat([df_res,
                   pd.DataFrame([[initparam, n_initparam, max_iterparam, tolparam, datasets[4], missed]],
                                columns=res_cols)])


#
#-- Sort the results by the number of missed predictions in the ascending order
df_res = df_res.sort_values(by=['missed', 'max_iter', 'tol'],
                            ascending=[True, True, False])
df_res = df_res.reset_index().rename(columns={'index': 'Index'})
print(f'--- df_res: \n{df_res}')


#
#-- Write the datasets and predicted results to the Excel file with ExcelWriter

# Original dataset and results
df_output = dfo.copy()
df_output = df_output.reset_index().rename(columns={'index': 'Index'})
df_output['Class Id'] = targets.reshape(150, 1)
df_output['Class Name'] = target_names[targets].reshape(150, 1)
df_output['ClassOrig'] = y_predo.reshape(150, 1)

# Standardized dataset and results
dfs = dfs.rename(columns={'Sepal Length': 'Sepal Length(STD)',
                          'Sepal Width': 'Sepal Width(STD)',
                          'Petal Length': 'Petal Length(STD)',
                          'Petal Width': 'Petal Width(STD)'})
dfs = dfs.reset_index().rename(columns={'index': 'Index'})
df_output = pd.merge(df_output, dfs, on='Index')
df_output['ClassSTD'] = y_preds.reshape(150, 1)

# Log calculated dataset and results
dfl = dfl.rename(columns={'Sepal Length': 'Sepal Length(Log10)',
                          'Sepal Width': 'Sepal Width(Log10)',
                          'Petal Length': 'Petal Length(Log10)',
                          'Petal Width': 'Petal Width(Log10)'})
dfl = dfl.reset_index().rename(columns={'index': 'Index'})
df_output = pd.merge(df_output, dfl, on='Index')
df_output['ClassLog10'] = y_predl.reshape(150, 1)

# Principal components and results
dfp = dfp.reset_index().rename(columns={'index': 'Index'})
df_output = pd.merge(df_output, dfp, on='Index')
df_output['ClassPCA'] = y_predp.reshape(150, 1)

# Results of the dataset with only petal features
df_output['ClassPetalOnly'] = y_predpetal.reshape(150, 1)

print(df_output.head())

# File name and sheet name. If the sheet exists, replace it
file_path = "Iris.xlsx"
sheetname = 'IrisClusters'
with pd.ExcelWriter(file_path,
                    engine='openpyxl',
                    mode='a',
                    if_sheet_exists='replace') as writer:
    df_output.to_excel(writer, sheet_name=sheetname, index=False)
