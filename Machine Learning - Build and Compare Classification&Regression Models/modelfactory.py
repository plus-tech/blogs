import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

class ClassifierFactory():
    __dtc = 'DecisionTreeClassifier'
    __gbc = 'GradientBoostingClassifier'
    __lrc = 'LogisticRegression'
    __rfc = 'RandomForestClassifier'
    __svc = 'SVC'
    __svl = 'LinearSVC'

    @property
    def dtc(self):
        return self.__dtc

    @property
    def gbc(self):
        return self.__gbc

    @property
    def lrc(self):
        return self.__lrc

    @property
    def rfc(self):
        return self.__rfc

    @property
    def svc(self):
        return self.__svc

    @property
    def svl(self):
        return self.__svl

    def newclassifier(self, name, params):
        """
        Acting as a factory method, new a classification model with the parameters specified by params.

        :param name: str,
            Name of the model to be instantiated.

        :param params: pandas.DataFrame,
            Parameters passed to the model in the creation.
            Contains at least Name and Parameter and Value columns.

        :return:
            model: object, Instance of the model |
            desc: str, description of the model
        """
        # random_state for DecisionTreeClassifier, GradientBoostingClassifier
        #    RandomForestClassifier, SVC, LinearSVC
        random_state = 0
        # max_depth for DecisionTreeClassifier
        max_depth = None
        # learning_rate for GradientBoostingClassifier
        learning_rate = 0.2
        # n_estimators for RandomForestClassifier
        n_estimators = 100
        # max_iter for LinearSVC
        max_iter = 5000

        # DecisionTreeClassifier
        if name == self.dtc:
            df_dtc = params.loc[params.Name == self.dtc, :]

            if len(df_dtc) > 0:
                if 'max_depth' in df_dtc.Parameter.unique():
                    max_depth = int(df_dtc.query("Parameter=='max_depth'").Value.tolist()[0])
                if 'random_state' in df_dtc.Parameter.unique():
                    random_state = int(df_dtc.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(max_depth={max_depth}, random_state={random_state})'

            model = DecisionTreeClassifier(max_depth=max_depth,
                                           random_state=random_state
                                           )
        # GradientBoostingClassifier
        elif name == self.gbc:
            df_gbc = params.loc[params.Name == self.gbc, :]

            if len(df_gbc) > 0:
                if 'random_state' in df_gbc.Parameter.unique():
                    random_state = int(df_gbc.query("Parameter=='random_state'").Value.tolist()[0])
                if 'learning_rate' in df_gbc.Parameter.unique():
                    learning_rate = float(df_gbc.query("Parameter=='learning_rate'").Value.tolist()[0])

            desc = f'{name}(random_state={random_state}, learning_rate={learning_rate})'

            model = GradientBoostingClassifier(random_state=random_state,
                                               learning_rate=learning_rate
                                               )
        # LogisticRegression
        elif name == self.lrc:
            df_lrc = params.loc[params.Name == self.lrc, :]

            if len(df_lrc) > 0:
                if 'random_state' in df_lrc.Parameter.unique():
                    random_state = int(df_lrc.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(random_state={random_state})'

            model = LogisticRegression(random_state=random_state
                                       )
        # RandomForestClassifier
        elif name == self.rfc:
            df_rfc = params.loc[params.Name == self.rfc, :]

            if len(df_rfc) > 0:
                if 'n_estimators' in df_rfc.Parameter.unique():
                    n_estimators = int(df_rfc.query("Parameter=='n_estimators'").Value.tolist()[0])
                if 'random_state' in df_rfc.Parameter.unique():
                    random_state = int(df_rfc.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(n_estimators={n_estimators}, random_state={random_state})'

            model = RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=random_state
                                           )
        # SVC
        elif name == self.svc:
            df_svc = params.loc[params.Name == self.svc, :]

            if len(df_svc) > 0:
                if 'random_state' in df_svc.Parameter.unique():
                    random_state = int(df_svc.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(random_state={random_state})'

            model = SVC(random_state=random_state
                        )
        # LinearSVC
        elif name == self.svl:
            df_svl = params.loc[params.Name == self.svl, :]

            if len(df_svl) > 0:
                if 'random_state' in df_svl.Parameter.unique():
                    random_state = int(df_svl.query("Parameter=='random_state'").Value.tolist()[0])
                if 'max_iter' in df_svl.Parameter.unique():
                    max_iter = int(df_svl.query("Parameter=='max_iter'").Value.tolist()[0])

            desc = f'{name}(random_state={random_state}, max_iter={max_iter})'

            model = LinearSVC(random_state=random_state,
                              max_iter=max_iter
                              )
        # Undefined
        else:
            desc = f'{name} is not implemented as of now'
            model = None

        return model, desc

    def execute(self, model, x_train, y_train, x_test, y_test):
        """
        Calls fit and predict and score on the model.
        Computes accuracy score for the training data and the test data

        :param model: instance of the model
        :param x_train: the training features
        :param y_train: the training labels
        :param x_test: the test features
        :param y_test: the test labels
        :return: score_model: float, score of the test data |
            acc_train: float, accuracy score of the training data |
            acc_test: float, accuracy score of the test data
        """
        score_model = None
        acc_train = None
        acc_test = None

        if model is not None:
            # learning
            model.fit(x_train, y_train)

            # predict
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            # evaluate
            score_model = model.score(x_test, y_test)    # accuracy score of test data
            acc_train = accuracy_score(y_train, y_train_pred)
            acc_test = accuracy_score(y_test, y_test_pred)

        return score_model, acc_train, acc_test

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class RegressorFactory():
    __dtr = 'DecisionTreeRegressor'
    __gbr = 'GradientBoostingRegressor'
    __las = 'Lasso'
    __mtp = 'Multiple'
    __ply = 'Polynomial'
    __rfr = 'RandomForestRegressor'
    __rdg = 'Ridge'
    __svr = 'SVR(rbf)'
    __svl = 'SVR(linear)'

    @property
    def dtr(self):
        return self.__dtr

    @property
    def gbr(self):
        return self.__gbr

    @property
    def las(self):
        return self.__las

    @property
    def mtp(self):
        return self.__mtp

    @property
    def ply(self):
        return self.__ply

    @property
    def rfr(self):
        return self.__rfr

    @property
    def rdg(self):
        return self.__rdg

    @property
    def svr(self):
        return self.__svr

    @property
    def svl(self):
        return self.__svl

    def newregressor(self, name, params):
        """
        Acting as a factory method, new a regression model with the parameters specified by params.

        :param name: str,
            Name of the model to be instantiated.

        :param params: pandas.DataFrame,
            Parameters passed to the model in the creation.
            Contains at least Name and Parameter and Value columns.

        :return:
            model: object, Instance of the model |
            desc: str, description of the model
        """
        # random_state for DecisionTreeRegressor, GradientBoostingRegressor
        #     RandomForestRegressor, Lasso, Ridge
        random_state = 0
        # max_depth for DecisionTreeRegressor
        max_depth = None
        # learning_rate for GradientBoostingRegressor
        learning_rate = 0.1
        # n_estimators for GradientBoostingRegressor, RandomForestRegressor
        n_estimators = 100
        # C for SVF
        c = 1.0
        # alpha for Lasso
        alpha = 1.0

        # DecisionTreeRegressor
        if name == self.dtr:
            df_dtr = params.loc[params.Name == self.dtr, :]

            if len(df_dtr) > 0:
                if 'max_depth' in df_dtr.Parameter.unique():
                    max_depth = int(df_dtr.query("Parameter=='max_depth'").Value.tolist()[0])
                if 'random_state' in df_dtr.Parameter.unique():
                    random_state = int(df_dtr.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(max_depth={max_depth}, random_state={random_state})'

            model = DecisionTreeRegressor(max_depth=max_depth,
                                          random_state=random_state
                                          )
        # GradientBoostingRegressor
        elif name == self.gbr:
            df_gbr = params.loc[params.Name == self.gbr, :]

            if len(df_gbr) > 0:
                if 'random_state' in df_gbr.Parameter.unique():
                    random_state = int(df_gbr.query("Parameter=='random_state'").Value.tolist()[0])
                if 'learning_rate' in df_gbr.Parameter.unique():
                    learning_rate = float(df_gbr.query("Parameter=='learning_rate'").Value.tolist()[0])
                if 'n_estimators' in df_gbr.Parameter.unique():
                    n_estimators = int(df_gbr.query("Parameter=='n_estimators'").Value.tolist()[0])

            desc = f'{name}(random_state={random_state}, learning_rate={learning_rate}, n_estimators={n_estimators})'

            model = GradientBoostingRegressor(random_state=random_state,
                                              learning_rate=learning_rate,
                                              n_estimators=n_estimators
                                              )
        # Lasso
        elif name == self.las:
            df_las = params.loc[params.Name == self.las, :]

            if len(df_las) >= 0:
                if 'alpha' in df_las.Parameter.unique():
                    alpha = float(df_las.query("Parameter=='alpha'").Value.tolist()[0])
                if 'random_state' in df_las.Parameter.unique():
                    random_state = int(df_las.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(alpha={alpha}, random_state={random_state})'

            model = Lasso(alpha=alpha,
                          random_state=random_state
                          )
        # Multiple
        elif name == self.mtp:
            df_mtp = params.loc[params.Name == self.mtp, :]

            if len(df_mtp.Parameter.unique()) > 0:
                pass

            desc = f'{name}()'

            model = LinearRegression()
        # Polynomial
        elif name == self.ply:
            df_ply = params.loc[params.Name == self.ply, :]

            if len(df_ply) >= 0:
                pass

            desc = f'{name}()'

            model = LinearRegression()
        # RandomForestRegressor
        elif name == self.rfr:
            df_rfr = params.loc[params.Name == self.rfr, :]

            if len(df_rfr) > 0:
                if 'n_estimators' in df_rfr.Parameter.unique():
                    n_estimators = int(df_rfr.query("Parameter=='n_estimators'").Value.tolist()[0])
                if 'random_state' in df_rfr.Parameter.unique():
                    random_state = int(df_rfr.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(n_estimators={n_estimators}, random_state={random_state})'

            model = RandomForestRegressor(n_estimators=n_estimators,
                                          random_state=random_state
                                          )
        # Ridge
        elif name == self.rdg:
            df_rdg = params.loc[params.Name == self.rdg, :]

            if len(df_rdg) >= 0:
                if 'alpha' in df_rdg.Parameter.unique():
                    alpha = float(df_rdg.query("Parameter=='alpha'").Value.tolist()[0])
                if 'random_state' in df_rdg.Parameter.unique():
                    random_state = int(df_rdg.query("Parameter=='random_state'").Value.tolist()[0])

            desc = f'{name}(alpha={alpha}, random_state={random_state})'

            model = Ridge(alpha=alpha,
                          random_state=random_state
                          )
        # SVR(rbf)
        elif name == self.svr:
            df_svr = params.loc[params.Name == self.svr, :]

            if len(df_svr) > 0:
                if 'C' in df_svr.Parameter.unique():
                    c = float(df_svr.query("Parameter=='C'").Value.tolist()[0])

            desc = f"{name}(C={c}, kernel='rbf')"

            model = SVR(C=c,
                        kernel='rbf'
                        )
        # SVR(linear)
        elif name == self.svl:
            df_svl = params.loc[params.Name == self.svl, :]

            if len(df_svl) > 0:
                if 'C' in df_svl.Parameter.unique():
                    c = float(df_svl.query("Parameter=='C'").Value.tolist()[0])

            desc = f"{name}(C={c}, kernel='linear')"

            model = SVR(C=c,
                        kernel='linear'
                        )
        # undefined
        else:
            desc = f'{name} is not implemented as of now'
            model = None

        return model, desc

    def execute(self, model, x_train, y_train, x_test, y_test):
        """
        Calls fit and predict and score on the model.
        Computes root mean squared errors (RMSE) for the training data and the test data

        :param model: instance of the model
        :param x_train: the training features
        :param y_train: the training labels
        :param x_test: the test features
        :param y_test: the test labels
        :return: score_model: float, score of the test data, |
            rmse_train: float, RMSE of the training data, |
            rmse_test: float, RMSE of the test data
        """
        score_model = None
        rmse_train = None
        rmse_test = None

        if model is not None:
            # learning
            model.fit(x_train, y_train)

            # predict
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            # evaluate
            score_model = model.score(x_test, y_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        return score_model, rmse_train, rmse_test