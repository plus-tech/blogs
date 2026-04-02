from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Iris dataset
iris = datasets.load_iris()

features, targets = iris.data, iris.target
feature_names = np.char.title(np.char.replace(iris.feature_names, ' (cm)', ''))
target_names = np.char.title(iris.target_names)
bins = 50
dfiris = pd.DataFrame(features, columns=feature_names)
dfiris.hist(bins=bins)
plt.show()

# ifprepnewdata = True
ifprepnewdata = False
# No. of cross validation folds
pcv = 5
# Random state for the data split function and the models
prandom_state = 0

# Shuffle the features and the targets
rng = np.random.default_rng(seed=prandom_state)
indices = rng.permutation(features.shape[0])
shuffled_features = features[indices]
shuffled_targets = targets[indices]

if ifprepnewdata:
    # Ratio 70/15/15 is used to split the dataset into 3 sets
    # 128 extracted from the original dataset as the input
    # 128*0.8204 = 105 as the training dataset
    # 128*0.1796 = 23 as the test dataset
    # 22 left as the new data, further validates the trained model
    # 5 as CV in GridSearchCV

    inputs = shuffled_features[0:128]
    inputsnew = shuffled_features[128:150]

    labels = shuffled_targets[0:128]
    labelsnew = shuffled_targets[128:150]

    ptest_size = 0.1796
else:
    inputs, labels = shuffled_features, shuffled_targets
    inputsnew, labelsnew = [], []

    ptest_size = 0.2

#
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    inputs,
    labels,
    test_size=ptest_size,
    random_state=prandom_state)

strsetsize = f'{len(X_train)} allocated for training; '
strsetsize += f'{len(X_test)} allocated for test; '
strsetsize += f'{len(inputsnew)} allocated for validation'
print(strsetsize)

# For data preprocess, it can deal with ['NONE', 'LOG1P', 'STD', 'BOX']
for preprocess in ['NONE', 'STD', 'BOX']:
    #
    # Use the orginal features
    if preprocess == 'NONE':
        xp_train, xp_test = X_train, X_test
        if ifprepnewdata:
            xp_new = inputsnew
    #
    # Standardization
    elif preprocess == 'STD':
        sc = StandardScaler()
        xp_train = sc.fit_transform(X_train)
        xp_test = sc.transform(X_test)

        if ifprepnewdata:
            xp_new = sc.transform(inputsnew)
    #
    # Box-Cox transformation
    elif preprocess == 'BOX':
        pt = PowerTransformer(method='box-cox', standardize=True)
        xp_train = pt.fit_transform(X_train)
        xp_test = pt.transform(X_test)

        if ifprepnewdata:
            xp_new = pt.transform(inputsnew)
    #
    # Log1p transformation
    elif preprocess == 'LOG1P':
        xp_train, xp_test = np.log1p(X_train), np.log1p(X_test)

        if ifprepnewdata:
            xp_new = np.log1p(inputsnew)
    else:
        raise Exception(f'{preprocess} preprocess is not supported yet.')

    # Concatenate the entire dataset after transformed
    xp = np.vstack([xp_train, xp_test])
    y = np.hstack([y_train, y_test])

    print(f'\nPreprocess: {preprocess}')
    dfp = pd.DataFrame(xp, columns=feature_names)
    # dfp.hist(bins=bins)
    # plt.show()

    # Initialize the classifier and the grid search
    #   - estimator: The model to tune
    #   - param_grid: The grid of parameters
    #   - cv: Number of cross-validation folds
    #   - scoring: The evaluation metric
    #   - verbose: Controls the output detail

    print(f"\nSVC ---------------")
    # Define a grid of hyperparameters
    pg_svc = {
        'kernel': ['rbf', 'linear', 'sigmoid'],
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 3, 7),
        'max_iter': [1000, 1200, 1500, 1800, 2000, 2500]
    }

    # Support Vector Machine Classifier (SVC)
    model_svc = SVC(random_state=prandom_state)
    grid_svc = GridSearchCV(
        estimator=model_svc,
        param_grid=pg_svc,
        cv=pcv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2)
    # Fit the grid search to the data
    grid_svc.fit(xp_train, y_train)

    # View the information of the best estimator
    print(f"Best parameters found: {grid_svc.best_params_}")
    print(f"Best estimator found: {grid_svc.best_estimator_}")
    print(f"Best cross-validation score: {grid_svc.best_score_}")

    # Evaluate the optimized model on the test set
    best_modelsvc = grid_svc.best_estimator_
    y_test_pred = best_modelsvc.predict(xp_test)
    acc_testsvc = accuracy_score(y_test, y_test_pred)
    print(f'accuracy score of test dataset: {acc_testsvc}')

    # Evaluate the model on the training dataset
    y_train_pred = best_modelsvc.predict(xp_train)
    acc_trainsvc = accuracy_score(y_train, y_train_pred)
    print(f'accuracy score of training dataset: {acc_trainsvc}')

    # Evaluate the model on the entire dataset
    y_predsvc = best_modelsvc.predict(xp)
    print(f'accuracy score of entire dataset: {accuracy_score(y, y_predsvc)}')

    if ifprepnewdata:
        y_new_pred = best_modelsvc.predict(xp_new)
        print(f'accuracy score of new dataset: {accuracy_score(labelsnew, y_new_pred)}')

    # LinearSVC
    print(f"\nLinearSVC ---------------")
    pg_linsvc = {
        'dual': ['auto'],
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2'],
        'loss': ['squared_hinge'],
        'max_iter': [10000, 15000, 20000]
    }

    model_linsvc = LinearSVC(random_state=prandom_state)
    grid_linsvc = GridSearchCV(
        estimator=model_linsvc,
        param_grid=pg_linsvc,
        return_train_score=True,
        cv=pcv,
        n_jobs=-1,
        verbose=2)

    # Fit the grid search to the data
    grid_linsvc.fit(xp_train, y_train)

    # View the information of the best estimator
    print(f"Best parameters found: {grid_linsvc.best_params_}")
    print(f"Best estimator found: {grid_linsvc.best_estimator_}")
    print(f"Best cross-validation score: {grid_linsvc.best_score_}")

    # Evaluate the optimized model on the test set
    best_modellinsvc = grid_linsvc.best_estimator_
    y_test_pred = best_modellinsvc.predict(xp_test)
    acc_testlinsvc = accuracy_score(y_test, y_test_pred)
    print(f'accuracy score of test dataset: {acc_testlinsvc}')

    # Evaluate the model on the training dataset
    y_train_pred = best_modellinsvc.predict(xp_train)
    acc_trainlinsvc = accuracy_score(y_train, y_train_pred)
    print(f'accuracy score of training dataset: {acc_trainlinsvc}')

    # Evaluate the model on the entire dataset
    y_predlinsvc = best_modellinsvc.predict(xp)
    print(f'accuracy score of entire dataset: {accuracy_score(y, y_predlinsvc)}')

    if ifprepnewdata:
        y_new_pred = best_modellinsvc.predict(xp_new)
        print(f'accuracy score of new dataset: {accuracy_score(labelsnew, y_new_pred)}')

    #
    # LogisticRegression
    print(f"\nLogisticRegression ---------------")
    pg_logi = {
        'penalty': [None, 'l2'],
        'C': np.logspace(-3, 3, 7),
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced'],
        'max_iter': [500, 1000, 1500, 2000]
    }
    model_logi = LogisticRegression(random_state=prandom_state)
    grid_logi = GridSearchCV(
        estimator=model_logi,
        param_grid=pg_logi,
        return_train_score=True,
        cv=pcv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2)
    # Fit the grid search to the data
    grid_logi.fit(xp_train, y_train)

    # View the information of the best estimator
    print(f"Best parameters found: {grid_logi.best_params_}")
    print(f"Best estimator found: {grid_logi.best_estimator_}")
    print(f"Best cross-validation score: {grid_logi.best_score_}")

    # Evaluate the optimized model on the test set
    best_modellogi = grid_logi.best_estimator_
    y_test_pred = best_modellogi.predict(xp_test)
    acc_testlogi = accuracy_score(y_test, y_test_pred)
    print(f'accuracy score of test dataset: {acc_testlogi}')

    # Evaluate the model on the training dataset
    y_train_pred = best_modellogi.predict(xp_train)
    acc_trainlogi = accuracy_score(y_train, y_train_pred)
    print(f'accuracy score of training dataset: {acc_trainlogi}')

    # Evaluate the model on the entire dataset
    y_predlogic = best_modellogi.predict(xp)
    print(f'accuracy score of entire dataset: {accuracy_score(y, y_predlogic)}')

    if ifprepnewdata:
        y_new_pred = best_modellogi.predict(xp_new)
        print(f'accuracy score of new dataset: {accuracy_score(labelsnew, y_new_pred)}')

    # Export the search results, associated with SVC and LogisticRegression
    X = np.vstack([X_train, X_test])
    dfsvc = pd.DataFrame(X, columns=feature_names)
    dfsvc['Class'] = y[:, None]
    classname = target_names[y]
    dfsvc['Class Name'] = classname[:, None]

    dflogi = dfsvc.copy()

    dfsvc['Pred'] = y_predsvc[:, None]
    dfsvc['Best Score'] = np.tile(grid_svc.best_score_, (len(y), 1))
    dfsvc['Train Score'] = np.tile(acc_trainsvc, (len(y), 1))
    dfsvc['Test Score'] = np.tile(acc_testsvc, (len(y), 1))
    dfsvc['Index'] = dfsvc.index

    dflogi['Pred'] = y_predlogic[:, None]
    dflogi['Best Score'] = np.tile(grid_logi.best_score_, (len(y), 1))
    dflogi['Train Score'] = np.tile(acc_trainlogi, (len(y), 1))
    dflogi['Test Score'] = np.tile(acc_testlogi, (len(y), 1))
    dflogi['Index'] = dflogi.index

    # Replace the sheet if it exists
    file_path = "Iris_Classification.xlsx"
    sheetsvc = 'BestSVC_' + preprocess
    sheetlogi = 'BestLogi_' + preprocess
    with pd.ExcelWriter(file_path,
                        engine='openpyxl',
                        mode='a',
                        if_sheet_exists='replace') as writer:
        dfsvc.to_excel(writer, sheet_name=sheetsvc, index=False)
        dflogi.to_excel(writer, sheet_name=sheetlogi, index=False)
