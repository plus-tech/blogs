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

    def newclassifier(self, name, hparams):
        """
        Acting as a factory method, new a classification model with the hyperparameters specified by hparams.

        :param name: str,
            Name of the model to be instantiated.

        :param hparams: dictionary,
            Hyperparameters passed to the model at the creation.
            Contains pairs of hyperparameter names and values.

        :return:
            model: object, Instance of the model |
            desc: str, description of the model
        """

        strcriterion = 'criterion'
        strmax_depth = 'max_depth'
        strrandom_state = 'random_state'

        strn_estimators = 'n_estimators'
        strlearning_rate = 'learning_rate'

        strmax_iter = 'max_iter'

        # SVC
        strkernel = 'kernel'

        # LinearSVC
        strdual = 'dual'

        # DecisionTreeClassifier
        if name == self.dtc:
            pcriterion = 'gini'
            pmax_depth = None
            prandom_state = None

            if strcriterion in hparams:
                pcriterion = hparams.get(strcriterion)
                # pcriterion = hparams[strcriterion]
            if strmax_depth in hparams:
                pmax_depth = hparams.get(strmax_depth)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(criterion={pcriterion}, max_depth={pmax_depth}, random_state={prandom_state})'

            model = DecisionTreeClassifier(criterion=pcriterion,
                                           max_depth=pmax_depth,
                                           random_state=prandom_state
                                           )
        # GradientBoostingClassifier
        elif name == self.gbc:
            pn_estimators = 100
            plearning_rate = 0.1
            prandom_state = None

            if strn_estimators in hparams:
                pn_estimators = hparams.get(strn_estimators)
            if strlearning_rate in hparams:
                plearning_rate = hparams.get(strlearning_rate)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(learning_rate={plearning_rate}, n_estimators={pn_estimators}, random_state={prandom_state}, )'

            model = GradientBoostingClassifier(learning_rate=plearning_rate,
                                               n_estimators=pn_estimators,
                                               random_state=prandom_state
                                               )
        # LogisticRegression
        elif name == self.lrc:
            pmax_iter = 100
            prandom_state = None

            if strmax_iter in hparams:
                pmax_iter = hparams.get(strmax_iter)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(max_iter={pmax_iter}, random_state={prandom_state})'

            model = LogisticRegression(max_iter=pmax_iter,
                                       random_state=prandom_state
                                       )
        # RandomForestClassifier
        elif name == self.rfc:
            pn_estimators = 100
            prandom_state = None

            if strn_estimators in hparams:
                pn_estimators = hparams.get(strn_estimators)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(n_estimators={pn_estimators}, random_state={prandom_state})'

            model = RandomForestClassifier(n_estimators=pn_estimators,
                                           random_state=prandom_state
                                           )
        # Support Vector Machine Classifier (SVC)
        elif name == self.svc:
            pkernel = 'rbf'
            pmax_iter = -1
            prandom_state = None

            if strkernel in hparams:
                pkernel = hparams.get(strkernel)
            if strmax_iter in hparams:
                pmax_iter = hparams.get(strmax_iter)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(kernel={pkernel}, max_iter={pmax_iter}, random_state={prandom_state})'

            model = SVC(kernel=pkernel,
                        max_iter=pmax_iter,
                        random_state=prandom_state
                        )
        # LinearSVC
        elif name == self.svl:
            pdual = 'auto'
            pmax_iter = 1000
            prandom_state = None

            if strdual in hparams:
                pdual = hparams.get(strdual)
            if strmax_iter in hparams:
                pmax_iter = hparams.get(strmax_iter)
            if strrandom_state in hparams:
                prandom_state = hparams.get(strrandom_state)

            desc = f'{name}(dual={pdual}, max_iter={pmax_iter}, random_state={prandom_state})'

            model = LinearSVC(random_state=prandom_state,
                              max_iter=pmax_iter,
                              dual=pdual
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
            acc_test: float, accuracy score of the test data |
            acc_all: float, accuracy score of the entire data
        """
        score_model = None
        acc_train = None
        acc_test = None
        acc_all = None

        if model is not None:
            # learning
            model.fit(x_train, y_train)

            # predict
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            # evaluate the trained model on the entire dataset
            x = np.vstack([x_train, x_test])
            y = np.hstack([y_train, y_test])
            y_pred = model.predict(x)

            # evaluate
            score_model = model.score(x_test, y_test)  # the same as the score of the test data
            acc_train = accuracy_score(y_train, y_train_pred)
            acc_test = accuracy_score(y_test, y_test_pred)
            acc_all = accuracy_score(y, y_pred)

        return score_model, acc_train, acc_test, acc_all