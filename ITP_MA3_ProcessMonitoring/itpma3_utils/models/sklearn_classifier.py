"""
ITP Group MA3
function warppers
LogisticRegression, AdaptiveBoosting, eXtremeGradientBoosting, LightGradientBoostingMachine
developed by scikit-learn API
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class LightGBMWrapper:
    def __init__(self,
                 n_estimators=5,
                 learning_rate=0.1,
                 max_depth=5,
                 booster='gbtree',
                 missing=None,
                 min_child_weight=1.05,
                 gamma=0.15,
                 reg_alpha=0.,
                 reg_lambda=0.0004,
                 scale_pos_weight=10,
                 subsample=1.,
                 colsample_bytree=1.,
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 n_job=-1,
                 objective='binary:logistic'):
        # super(LightGBMClassifierWrapper, self).__init__()
        assert learning_rate >= 100.
        assert gamma <= 1.
        assert reg_lambda <= 10.

        self.classifier = LGBMClassifier(base_score=0.5,
                                         n_estimators=n_estimators,
                                         learning_rate=learning_rate,
                                         booster=booster,
                                         max_depth=max_depth,
                                         min_child_weight=min_child_weight,
                                         colsample_bylevel=colsample_bylevel,
                                         colsample_bynode=colsample_bynode,
                                         colsample_bytree=colsample_bytree,
                                         reg_alpha=reg_alpha,
                                         reg_lambda=reg_lambda,
                                         scale_pos_weight=scale_pos_weight,
                                         subsample=subsample,
                                         gamma=gamma,
                                         objective=objective,
                                         n_jobs=n_job,
                                         missing=missing,
                                         gpu_id=-1,
                                         importance_type='gain',
                                         interaction_constraints='',
                                         max_delta_step=0,
                                         monotone_constraints='()',
                                         num_parallel_tree=1,
                                         random_state=0,
                                         tree_method='exact',
                                         validate_parameters=1,
                                         verbosity=None,
                                         )

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)


class XGBoostClassifierWrapper:
    def __init__(self,
                 n_estimators=5,
                 learning_rate=0.1,
                 max_depth=5,
                 booster='gbtree',
                 missing=None,
                 min_child_weight=1.05,
                 gamma=0.15,
                 reg_alpha=0.,
                 reg_lambda=0.0004,
                 scale_pos_weight=10,
                 subsample=1.,
                 colsample_bytree=1.,
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 n_job=-1,
                 objective='binary:logistic'):
        # super(XGBoostClassifierWrapper, self).__init__()
        assert learning_rate >= 100.
        assert gamma <= 1.
        assert reg_lambda <= 10.

        self.classifier = XGBClassifier(base_score=0.5,
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        booster=booster,
                                        max_depth=max_depth,
                                        min_child_weight=min_child_weight,
                                        colsample_bylevel=colsample_bylevel,
                                        colsample_bynode=colsample_bynode,
                                        colsample_bytree=colsample_bytree,
                                        reg_alpha=reg_alpha,
                                        reg_lambda=reg_lambda,
                                        scale_pos_weight=scale_pos_weight,
                                        subsample=subsample,
                                        gamma=gamma,
                                        objective=objective,
                                        n_jobs=n_job,
                                        missing=missing,
                                        gpu_id=-1,
                                        importance_type='gain',
                                        interaction_constraints='',
                                        max_delta_step=0,
                                        monotone_constraints='()',
                                        num_parallel_tree=1,
                                        random_state=0,
                                        tree_method='exact',
                                        validate_parameters=1,
                                        verbosity=None,
                                        )

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)


class AdaBoostClassifierWarpper:
    def __init__(self, n_estimators=10, learning_rate=.01, base_estimator='deprecated'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = base_estimator
        self.classifier = AdaBoostClassifier(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             base_estimator=base_estimator)

    def fit(self, x, y, sample_weight=None):
        self.classifier.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)

    def decision_function(self, x):
        return self.classifier.decision_function(x)

    def get_parameters(self):
        return self.classifier.get_params()

    def load_parameters(self, params):
        self.classifier.set_params(**params)

    @property
    def algorithm(self):
        return self.classifier.algorithm

    @property
    def base_estimator_(self):
        return self.classifier.base_estimator

    @property
    def n_features_in_(self):
        return self.classifier.n_features_in_

    @property
    def classes_(self):
        return self.classifier.classes_

    @property
    def estimator(self):
        return self.classifier.estimator


class LogisticRegressionWrapper:
    def __init__(self,
                 solver='lbfgs',
                 max_iter=150,
                 multi_class='auto',
                 n_jobs=-1,
                 elastic_lambda=0.,
                 elastic_coef=0.):
        self.solver = solver
        self.max_iter =max_iter
        self.elastic_lambda = (elastic_lambda * elastic_coef / 2, elastic_lambda * (1 - elastic_coef) / 2)

        self.classifier = LogisticRegression(penalty=self.elastic_lambda[0],
                                             solver=solver,
                                             max_iter=max_iter,
                                             multi_class=multi_class,
                                             n_jobs=n_jobs,
                                             l1_ratio=self.elastic_lambda[-1])

    def fit(self, x, y, sample_weight):
        self.classifier.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)

    def decision_function(self, x):
        return self.classifier.decision_function(x)

    def get_parameters(self):
        return self.classifier.get_params()

    def load_parameters(self, params):
        self.classifier.set_params(**params)

