import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

ModelCols = ['Name', 'Category', 'PreCalc', 'Hyperparameter', 'Value']
#
# Model candidates and their default hyperparameters
ModelList = [
    ['DecisionTreeClassifier', 'Classification', 0, 'criterion', 'gini'],
    ['DecisionTreeClassifier', 'Classification', 0, 'max_depth', None],
    ['DecisionTreeClassifier', 'Classification', 0, 'random_state', 0],
    ['GradientBoostingClassifier', 'Classification', 0, 'learning_rate', 0.1],
    ['GradientBoostingClassifier', 'Classification', 0, 'random_state', 0],
    ['LogisticRegression', 'Classification', 0, 'max_iter', 5000],
    ['LogisticRegression', 'Classification', 0, 'random_state', 0],
    ['RandomForestClassifier', 'Classification', 0, 'n_estimators', 200],
    ['RandomForestClassifier', 'Classification', 0, 'random_state', 0],
    ['SVC', 'Classification', 0, 'kernel', 'rbf'],
    ['SVC', 'Classification', 0, 'random_state', 0],
    ['SVC', 'Classification', 0, 'max_iter', 2000],
    ['LinearSVC', 'Classification', 0, 'random_state', 0],
    ['LinearSVC', 'Classification', 0, 'max_iter', 2000],
]

df_model = pd.DataFrame(ModelList, columns=ModelCols)

def modelpredict(dfmodel, xtrain, xtest, ytrain, ytest):
    modellist = dfmodel.Name.unique().tolist()
    evalres = []

    #
    # Instantiate a ClassifierFactory
    mf = ClassifierFactory()

    for name in modellist:
        # Extract the hyperparameters for the specific model
        dfparam = dfmodel.loc[dfmodel.Name == name, :]
        hparam_dict = dict(zip(dfparam.Hyperparameter, dfparam.Value))
        # Create a model instance
        model, desc = mf.newclassifier(name, hparam_dict)

        if model is not None:
            # Fit and evaluate the model
            scoremodel, acctrain, acctest, accall = mf.execute(model, xtrain, ytrain, xtest, ytest)
        else:
            scoremodel, acctrain, acctest, accall = None, None, None, None

        # Append this particular model's execution results to the result list
        evalres.append([name, scoremodel, acctrain, acctest, accall])

    return evalres


# Load the dataset
iris = datasets.load_iris()

features, targets = iris.data, iris.target
feature_names = np.char.title(np.char.replace(iris.feature_names, ' (cm)', ''))
target_names = np.char.title(iris.target_names)
# print( iris.DESCR )

# Shuffle the features and the targets
indices = np.random.permutation(features.shape[0])
shuffled_features = features[indices]
shuffled_targets = targets[indices]

preprocess = ['None', 'Standardized', 'Log1p', 'Box-Coxed', 'PCA']
bins = 50

dfo = pd.DataFrame(shuffled_features, columns=feature_names)
print(dfo.describe())

#
# visualize the input data
print(f'\n{preprocess[0]}')
dfo.hist(bins=bins)
plt.show()

#
# default parameters for data split
pn_splits = 5  # Train-Test = 80/20
prandom_state = 0

# Configure KFold
kf = KFold(n_splits=pn_splits, shuffle=True, random_state=prandom_state)

eval_res = None
# Manually iterate through folds
for ite, (train_ind, test_ind) in enumerate(kf.split(shuffled_features)):
    ite += 1

    # Split the dataset into the training set and test set
    x_train, x_test = shuffled_features[train_ind], shuffled_features[test_ind]
    y_train, y_test = shuffled_targets[train_ind], shuffled_targets[test_ind]
    #
    # Preprocess: None, meaning using the original features
    evalo = modelpredict(df_model, x_train, x_test, y_train, y_test)
    tmpds = np.full((len(evalo), 1), preprocess[0])
    evalo = np.concatenate((evalo, tmpds), axis=1)

    #
    # Preprocess: Standardize the features
    sc = StandardScaler()
    xs_train = sc.fit_transform(x_train)
    xs_test = sc.transform(x_test)

    xs = np.vstack([xs_train, xs_test])
    dfs = pd.DataFrame(xs, columns=feature_names)
    # print(f'\n{preprocess[1]}')
    # dfs.hist(bins=bins)
    # plt.show()

    evals = modelpredict(df_model, xs_train, xs_test, y_train, y_test)
    tmpds = np.full((len(evals), 1), preprocess[1])
    evals = np.concatenate((evals, tmpds), axis=1)

    #
    # Preprocess: Transform the features with log1p function
    xl_train, xl_test = np.log1p(x_train), np.log1p(x_test)

    xl = np.vstack([xl_train, xl_test])
    dfl = pd.DataFrame(xl, columns=feature_names)
    # print(f'\n{preprocess[2]}')
    # dfl.hist(bins=bins)
    # plt.show()

    evall = modelpredict(df_model, xl_train, xl_test, y_train, y_test)
    tmpds = np.full((len(evals), 1), preprocess[2])
    evall = np.concatenate((evall, tmpds), axis=1)

    #
    # Preprocess: Box-Cox the features
    pt = PowerTransformer(method='box-cox', standardize=True)
    xb_train = pt.fit_transform(x_train)
    xb_test = pt.transform(x_test)

    xb = np.vstack([xb_train, xb_test])
    dfb = pd.DataFrame(xb, columns=feature_names)
    # print(f'\n{preprocess[3]}')
    # dfb.hist(bins=bins)
    # plt.show()

    evalb = modelpredict(df_model, xb_train, xb_test, y_train, y_test)
    tmpds = np.full((len(evalb), 1), preprocess[3])
    evalb = np.concatenate((evalb, tmpds), axis=1)

    #
    # Preprocess: Reduce dimensions using PCA
    pca = PCA(n_components=2)
    xp_train = pca.fit_transform(x_train)
    xp_test = pca.transform(x_test)
    xp = np.vstack([xp_train, xp_test])

    dfp = pd.DataFrame(xp, columns=['Component-1', 'Component-2'])
    # print(f'\n{preprocess[4]}')
    # dfp.hist(bins=bins)
    # plt.show()
    evalp = modelpredict(df_model, xp_train, xp_test, y_train, y_test)
    tmpds = np.full((len(evalp), 1), preprocess[4])
    evalp = np.concatenate((evalp, tmpds), axis=1)

    #
    # Consolidate the results at fold ite
    eval_ite = np.vstack([evalo, evals, evall, evalb, evalp])
    itercol = np.ones((len(eval_ite), 1), dtype=np.int16) * ite
    eval_ite = np.hstack([itercol, eval_ite])

    if eval_res is None:
        eval_res = eval_ite
    else:
        eval_res = np.vstack([eval_res, eval_ite])

# Format the execution results matrix using DataFrame
columns = ['Fold', 'Model', 'Score', 'Score(Train)', 'Score(Test)', 'Acc', 'Preprocess']
df_res = pd.DataFrame(eval_res, columns=columns)

df_res['Fold'] = df_res['Fold'].astype(int)

floatcols = ['Score', 'Score(Train)', 'Score(Test)', 'Acc']
df_res[floatcols] = df_res[floatcols].astype(float)
df_res[floatcols] = df_res[floatcols].round(3)

df_res['Index'] = df_res.index
df_res = df_res.sort_values(by=['Model', 'Preprocess', 'Acc'],
                            ascending=[True, True, False])

# Export the execution results to the Excel file, including all models and all folds
#   Replace the sheet if it exists
file_path = "Iris_Classification.xlsx"
sheetname = 'CompareClassifiers'
with pd.ExcelWriter(file_path,
                    engine='openpyxl',
                    mode='a',
                    if_sheet_exists='replace') as writer:
    df_res.to_excel(writer, sheet_name=sheetname, index=False)
