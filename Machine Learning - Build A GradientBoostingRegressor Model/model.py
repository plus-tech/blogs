# Gradient Boosting Regressor is used to predict mountain temperature
# in this example

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import oracledb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.style as plts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def PredictMountTemp():
    #
    # load the dataset from the database
    sqlstr = 'select * from mount_temp'
    columns = ['No', 'Mountain', 'Latitude', 'Longitude', 'Altitude', 'Max', 'Avg', 'Min', 'Prefecture']

    try:
        with oracledb.connect(user="test", password='1234', dsn="localhost/xepdb1") as conn:
            with conn.cursor() as cursor:
                dforg = pd.DataFrame(cursor.execute(sqlstr), columns=columns)
    except oracledb.Error as e:
        print(f'Failed to fetch data from the database ({str(e)})')
        return

    dforg = dforg.set_index('No')

    #
    # drop null data
    df = dforg.dropna()
    print(df[df["Avg"].isna()])

    #
    # extract the features and the target
    X = df[["Latitude", "Longitude", "Altitude"]]
    y = df["Avg"]

    print(pd.concat([X, y], axis=1))

    #
    # divide the dataset into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #
    # standardize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #
    # create the model
    gbrd = GradientBoostingRegressor(random_state=123)
    #
    # train the model
    gbrd.fit(X_train, y_train)
    #
    # make a prediction
    y_pred = gbrd.predict(X_test)
    #
    # evaluate the model
    scored = gbrd.score(X_test, y_test)
    #
    # save the predicted and actual temperatures
    tmp_pred = y_pred.reshape(np.size(y_pred), 1)
    tmp_test = y_test.values.reshape(np.size(y_test), 1)
    tempd = pd.DataFrame(np.hstack([tmp_test, tmp_pred]))
    rmsed_train = np.sqrt(mean_squared_error(y_train, gbrd.predict(X_train)))
    rmsed_test = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'rmse train: {rmsed_train}')
    print(f'rmse test: {rmsed_test}')

    #
    # using Grid Search to find the best estimator
    starttime = datetime.now()
    print('Grid search, starting from: ', starttime.isoformat())

    parameters = {
        'n_estimators' : [3, 5, 10, 30, 50, 100],
        'max_features' : [1, 3, 5, 10],
        'random_state' : [123],
        'min_samples_split' : [3, 5, 10, 30, 50],
        'max_depth' : [3, 5, 10, 30, 50]
    }
    gbrb = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameters, cv=10)
    gbrb.fit(X_train, y_train)
    #
    # time consumed
    endtime = datetime.now()
    print('Grid search, ending at: ', endtime.isoformat())
    print('Time consumed for optimization: ', (endtime-starttime))
    #
    # best estimator
    print('Best params: {0}'.format(gbrb.best_params_))
    print('Best estimator: {0}'.format(gbrb.best_estimator_))

    y_pred = gbrb.predict(X_test)

    scoreb = gbrb.score(X_test, y_test)
    #
    # save the predicted and actual temperatures
    tmp_pred = y_pred.reshape(np.size(y_pred), 1)
    tmp_test = y_test.values.reshape(np.size(y_test), 1)
    tempb = pd.DataFrame(np.hstack([tmp_test, tmp_pred]))
    rmseb_train = np.sqrt(mean_squared_error(y_train, gbrd.predict(X_train)))
    rmseb_test = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'rmse train: {rmseb_train}')
    print(f'rmse test: {rmseb_test}')

    #
    # visualize the results
    plts.use('ggplot')
    fig, ax = plt.subplots()
    #
    # default parameters
    ax.scatter(tempd.iloc[:, 0], tempd.iloc[:, 1], color='darkblue', label='default')
    #
    # optimized parameters
    ax.scatter(tempb.iloc[:, 0], tempb.iloc[:, 1], marker='x', color='crimson', label='optimized')
    #
    # reference
    xtmp = np.array([np.min(tempd.iloc[:, 1]), np.max(tempd.iloc[:, 1])])
    ytmp = xtmp.copy()
    ax.plot(xtmp, ytmp, label='reference', color='gray')
    #
    # show scores and correlation rates
    strd = 'score: ' + str(scored) + '   rmse: ' + str(rmsed_test)
    strb = 'score: ' + str(scoreb) + '   rmse: ' + str(rmseb_test)
    # strd = 'score: ' + str(round(scored, 2)) + '   corr: ' + str(round(corrd.iloc[0, 1], 2))
    # strb = 'score: ' + str(round(scoreb, 2)) + '   corr: ' + str(round(corrb.iloc[0, 1], 2))
    ax.text(xtmp.min(), ytmp.max(), strd, color='darkblue')
    ax.text(xtmp.min(), ytmp.max()-1,strb, color='crimson')
    #
    # graphical setting
    ax.legend(loc='lower right')
    ax.set_xlabel('actual temperature')
    ax.set_ylabel('predicted temperature')
    fig.suptitle('mountain temperature prediction')

    plt.show()
    #
    # save the trained model
    with open('MountTempModel.pkl', mode='wb') as f:
        pickle.dump(gbrb, f)


#
# test the saved mode
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

def CallPredictMountTemp():
    with open('MountTempModel.pkl', mode='rb') as fp:
        gbr = pickle.load(fp)

    xls = 'C:\\xForAll\\09_myBlog\\Python\\mount_temp.xlsx'
    df = pd.read_excel(xls, sheet_name='MountTemp') #, index_col='No')
    df = df.dropna()
    # df.sort_values(by='No', ascending=True)
    print(df)

    #
    # extract the features and the target
    X_test, y_test = df[["Latitude", "Longitude", "Altitude"]], df["Avg"]
    y_pred = gbr.predict(X_test)

    score = gbr.score(X_test, y_test)
    #
    # save the predicted and actual temperatures
    tmp_pred = y_pred.reshape(np.size(y_pred), 1)
    tmp_test = y_test.values.reshape(np.size(y_test), 1)
    temp = pd.DataFrame(np.hstack([tmp_pred, tmp_test]))
    print(temp)

    print('score is : {0}'.format(score))
    print('correlation rate is : \n{0}'.format(temp.corr()))

    temp = temp.rename(columns={0:"Prediction", 1:"Actual"})
    temp['No'] = np.arange(temp.Prediction.size)
    temp = temp.set_index('No')
    print(temp)
    temp.to_csv('PredictMountTemp_res.csv')
