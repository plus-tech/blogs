import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import oracledb

from modelfactory import ClassifierFactory

def CompareClassificationModels():
    """
    The function instantiates a series of classification models pre-defined in a table
    and fit the models and computes the predictions for a specific dataset
    and visualize the results.

    :parameter:
    No parameters are required.

    :raise:
    An oracledb.Error will be raised if anything goes wrong with the database.

    :return:
    This function doesn't return a value.
    """
    # ------------------------------------------------------------------
    # configure a logger
    # ------------------------------------------------------------------
    format = '%(asctime)s %(levelname)-10s [%(threadName)s] [%(module)s] [%(funcName)-30s] %(message)s'
    logger = logging.getLogger('modellogger')
    handler = logging.StreamHandler()
    fmt = logging.Formatter(format)
    handler.setFormatter(fmt=fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info('process starts')

    # ------------------------------------------------------------------
    # load the list of models and the dataset from the database
    # ------------------------------------------------------------------
    model_columns = ['Name', 'Category', 'PreCalc', 'Parameter', 'Value']
    sqlmodel = "select * from modellist where category='Classification'"
    sqldata = 'select * from datasetcls'

    try:
        with oracledb.connect(user="test", password='1234', dsn="localhost/xepdb1") as conn:
            with conn.cursor() as cursor:
                df_model = pd.DataFrame(cursor.execute(sqlmodel), columns=model_columns)
                df_data = pd.DataFrame(cursor.execute(sqldata))
    except oracledb.Error as e:
        logger.error(f'Failed to fetch data from the database ({str(e)})')
        return

    logger.debug(f"list of the models\n{df_model}")
    logger.debug(f"head of the dataset\n{df_data.head()}")

    # ------------------------------------------------------------------
    # data pre-processing
    # ------------------------------------------------------------------
    df_data = df_data.dropna()
    # visualize the input data
    # df_data.hist(bins=50)
    # plt.show()

    # separate the dataset into features and labels
    x, y = df_data.iloc[:, 0:-1], df_data.iloc[:, -1]
    logger.debug(f"the features\n{x}")
    logger.debug(f"the labels\n{y}")

    #
    # default parameters for dataset
    test_size = 0.3
    standardize = True
    random_state = 123

    # ------------------------------------------------------------------
    # prepare the training dataset and the test dataset and standardize
    # ------------------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if standardize:
        logger.info('Standardize the training and test datasets')
        sc = StandardScaler()
        x_train= sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

    # ------------------------------------------------------------------
    # create and fit and evaluate the models
    # ------------------------------------------------------------------
    model_list = df_model.Name.unique().tolist()
    eval_res = []
    if len(model_list) > 0:

        model_list.sort()
        mf = ClassifierFactory()

        for name in model_list:
            # create a model
            model, desc = mf.newclassifier(name, df_model)

            if model is not None:
                logger.info(desc)

                # execute fit, predict, evaluation
                score_model, acc_train, acc_test = mf.execute(model, x_train, y_train, x_test, y_test)

                eval_res.append([name, score_model, acc_train, acc_test])
                logger.info(f'  [Score: {score_model} Accuracy (Training): {acc_train} Accuracy (Test): {acc_test}]')
            else:
                logger.warning(desc)
                score_model, acc_train, acc_test = None, None, None
                logger.warning(f'  [Score: {score_model} Accuracy (Training): {acc_train} Accuracy (Test): {acc_test}]')

    # ------------------------------------------------------------------
    # visualize the evaluation results
    # ------------------------------------------------------------------
    if len(eval_res) > 0:
        logger.info('Visualize the evaluation results')

        columns = ['Model', 'Score', 'Accuracy (Training)', 'Accuracy (Test)']
        eval_res.sort()
        df = pd.DataFrame(eval_res, columns=columns)

        logger.debug(f"evaluation results\n{df}")

        plt.grid(which='major')

        #
        # reference
        xr = np.array([round(pd.DataFrame.min(df.iloc[:, 2:3])-0.05, 1), round(pd.DataFrame.max(df.iloc[:, 2:3])+0.05, 1)])
        yr = xr.copy()
        plt.plot(xr, yr, color='#D2D5D1')

        sns.scatterplot(
            data=df,
            x='Accuracy (Training)',
            y='Accuracy (Test)',
            marker='o',
            hue=df['Model']
        )

        plt.plot()

        dev = (xr.max() - xr.min()) / 100.0
        for eval in df.values:
            score = round(eval[1], 2)
            x = eval[2] + dev
            y= eval[3] + dev
            plt.text(x, y, score)

        plt.legend(loc='best')
        plt.suptitle('Compare Classification Models')
        plt.show()

    else:
        logger.info('No evaluation results are available for visualization')

    logger.info('process finishes')


import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import oracledb
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from modelfactory import RegressorFactory

def CompareRegressionModels():
    """
    The function instantiates a series of regression models pre-defined in a table
    and fit the models and computes the predictions for a specific dataset
    and visualize the results.

    :parameter:
    No parameters are required.

    :raise:
    An oracledb.Error will be raised if anything goes wrong with the database.

    :return:
    This function doesn't return a value.
    """
    # ------------------------------------------------------------------
    # configure a logger
    # ------------------------------------------------------------------
    format = '%(asctime)s %(levelname)-10s [%(threadName)s] [%(module)s] [%(funcName)-30s] %(message)s'
    logger = logging.getLogger('modellogger')
    handler = logging.StreamHandler()
    fmt = logging.Formatter(format)
    handler.setFormatter(fmt=fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info('process starts')

    # ------------------------------------------------------------------
    # load the list of models and the dataset from the database
    # ------------------------------------------------------------------
    model_columns = ['Name', 'Category', 'PreCalc', 'Parameter', 'Value']
    sqlmodel = "select * from modellist where category='Regression'"
    sqldata = 'select * from datasetreg'

    try:
        with oracledb.connect(user="test", password='1234', dsn="localhost/xepdb1") as conn:
            with conn.cursor() as cursor:
                df_model = pd.DataFrame(cursor.execute(sqlmodel), columns=model_columns)
                df_data = pd.DataFrame(cursor.execute(sqldata))
    except oracledb.Error as e:
        logger.error(f'Failed to fetch data from the database ({str(e)})')
        return

    logger.debug(f"list of the models\n{df_model}")
    logger.debug(f"head of the dataset\n{df_data.head()}")

    # ------------------------------------------------------------------
    # data pre-processing
    # ------------------------------------------------------------------
    df_data = df_data.dropna()
    # visualize the input data
    # df_data.hist(bins=50)
    # plt.show()

    # separate the dataset into features and labels
    x, y = df_data.iloc[:, 0:-1], df_data.iloc[:, -1]
    logger.debug(f"the features\n{x}")
    logger.debug(f"the labels\n{y}")

    #
    # default parameters for dataset
    test_size = 0.3
    standardize = True
    random_state = 123
    # degree for polynomial features
    degree = 3

    # ------------------------------------------------------------------
    # prepare the training dataset and the test dataset and standardize
    # ------------------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # standardize the input data
    logger.info('Standardize the training and test datasets')
    if standardize:
        sc = StandardScaler()
        x_train= sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

    # ------------------------------------------------------------------
    # prepare polynomially transformed data if needed
    # ------------------------------------------------------------------
    if df_model.query('PreCalc == 1').size > 0:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_train_ply = poly.fit_transform(x_train)
        x_test_ply = poly.fit_transform(x_test)

    # ------------------------------------------------------------------
    # create and fit and evaluate the models
    # ------------------------------------------------------------------
    model_list = df_model.Name.unique().tolist()
    model_list.sort()
    eval_res = []
    if len(model_list) > 0:
        mf = RegressorFactory()

        for name in model_list:
            # create a model
            model, desc = mf.newregressor(name, df_model)

            if model is not None:
                logger.info(desc)

                # execute fit, predict, evaluation
                if df_model.loc[df_model.Name == name, 'PreCalc'].unique()[0] > 0:
                    score_model, rmse_train, rmse_test = mf.execute(model, x_train_ply, y_train, x_test_ply, y_test)
                else:
                    score_model, rmse_train, rmse_test = mf.execute(model, x_train, y_train, x_test, y_test)

                eval_res.append([name, score_model, rmse_train, rmse_test])
                logger.info(f'  [Score: {score_model} RMSE (Training): {rmse_train} RMSE (Test): {rmse_test}]')
            else:
                logger.warning(desc)

                score_model, rmse_train, rmse_test = None, None, None
                logger.warning(f'  [Score: {score_model} RMSE (Training): {rmse_train} RMSE (Test): {rmse_test}]')

    # ------------------------------------------------------------------
    # visualize the evaluation results
    # ------------------------------------------------------------------
    if len(eval_res) > 0:
        logger.info('Visualize the evaluation results')

        columns = ['Model', 'Score', 'RMSE (Training)', 'RMSE (Test)']
        eval_res.sort()
        df = pd.DataFrame(eval_res, columns=columns)
        logger.debug(f"evaluation results\n{df}")

        plt.grid()
        #
        # reference
        xr = np.array([round(pd.DataFrame.min(df.iloc[:, 2:3])-0.05, 1), round(pd.DataFrame.max(df.iloc[:, 2:3])+0.05, 1)])
        yr = xr.copy()
        plt.plot(xr, yr, color='#D2D5D1')
        sns.scatterplot(
            data=df,
            x='RMSE (Training)',
            y='RMSE (Test)',
            marker='o',
            hue=df['Model']
        )

        dev = (xr.max() - xr.min()) / 100.0
        for eval in df.values:
            score = round(eval[1], 2)
            x = eval[2] + dev
            y= eval[3] + dev
            plt.text(x, y, score)

        plt.legend(loc='best')
        plt.suptitle('Compare Regression Models')
        plt.show()

    else:
        logger.info('No evaluation results are available for visualization')

    logger.info('process finishes')
    
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('-- Running at {0} --'.format(datetime.now().isoformat()))

    CompareClassificationModels()
    CompareRegressionModels()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/