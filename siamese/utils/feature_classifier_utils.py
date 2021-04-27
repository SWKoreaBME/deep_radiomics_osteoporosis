from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


# Feature Selection

def Lasso_feature_selection(X, y, tr=0.25):
    clf = LassoCV(cv=5)
    sfm = SelectFromModel(clf, threshold=tr)
    sfm.fit(X, y)
    return sfm


# Classifiers

def GridBackBone(clf, param_grid, X_train, y_train, cv):
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)

    return clf.best_params_


def GridRF(X_train, y_train, random_state, cv):
    # grid search CV
    print("Random Forest Grid search running...")

    rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # param_grid = {
    #     'n_estimators': [200, 300, 400, 500],
    #     'max_features': ['auto'],
    #     'max_depth': [8],
    #     'criterion': ['gini'],
    #     'class_weight': [None]
    # }

    param_grid = {
        'n_estimators': [200],
        'max_features': ['auto'],
        'max_depth' : [8],
        'criterion' :['gini'],
        'class_weight' : [None]
    }

    best_params = GridBackBone(clf=rfc, param_grid=param_grid, X_train=X_train, y_train=y_train, cv=cv)

    best_rf = RandomForestClassifier(random_state=random_state,
                                     max_features=best_params['max_features'],
                                     n_estimators=best_params['n_estimators'],
                                     max_depth=best_params['max_depth'],
                                     criterion=best_params['criterion'])

    print(best_params)

    return best_rf


def GridXGB(X_train, y_train, random_state, cv):
    # grid search CV
    print("XGBoost Grid search running...")

    #     rfc=RandomForestClassifier(random_state=random_state)
    xgbc = xgb.XGBClassifier(random_state=random_state, n_jobs=-1)

    # param_grid = {
    #     'n_estimators': [200, 500],
    #     'max_depth': [7, 8],
    #     'learning_rate': [0.1, 0.01],
    #     'booster': ['gbtree', ]
    # }
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'learning_rate': [0.1],
        'booster': ['gbtree']
    }

    best_params = GridBackBone(clf=xgbc, param_grid=param_grid, X_train=X_train, y_train=y_train, cv=cv)

    best_xgbc = xgb.XGBClassifier(random_state=random_state,
                                  learning_rate=best_params['learning_rate'],
                                  n_estimators=best_params['n_estimators'],
                                  max_depth=best_params['max_depth'],
                                  booster=best_params['booster'])

    print(best_params)
    return best_xgbc


def GridAda(X_train, y_train, random_state, cv):
    # grid search CV
    print("Adaboost Grid search running...")

    #     rfc=RandomForestClassifier(random_state=random_state)
    adbc = AdaBoostClassifier(random_state=random_state)

    # param_grid = {
    #     'n_estimators': [50, 100, 200, 300],
    #     'learning_rate': [1.0, 0.1, 0.01]
    # }
    param_grid = {
        'n_estimators': [300],
        'learning_rate' : [0.1]
    }

    best_params = GridBackBone(clf=adbc, param_grid=param_grid, X_train=X_train, y_train=y_train, cv=cv)

    best_adbc = AdaBoostClassifier(random_state=random_state,
                                   learning_rate=best_params['learning_rate'],
                                   n_estimators=best_params['n_estimators'])

    print(best_params)

    return best_adbc


def GridSVM(X_train, y_train, random_state, cv):
    # grid search CV
    print("Linear SVM Grid search running...")

    #     rfc=RandomForestClassifier(random_state=random_state)
    lsvc = LinearSVC(random_state=random_state)

    param_grid = {
        'penalty': ['l2'],
        'loss': ['hinge', 'squared_hinge'],
        'tol': [1e-06, 1e-05, 1e-04]
    }

    best_params = GridBackBone(clf=lsvc, param_grid=param_grid, X_train=X_train, y_train=y_train, cv=cv)

    best_lsvc = LinearSVC(random_state=random_state,
                          penalty=best_params['penalty'],
                          loss=best_params['loss'],
                          tol=best_params['tol'])

    print(best_params)

    return best_lsvc


def ensemble_voting(X_train, y_train, random_state, cv):
    print("Ensemble Voting Grid search running...")

    common_grid_params = [X_train, y_train, random_state, cv]

    rf = GridRF(*common_grid_params)
    xgb = GridXGB(*common_grid_params)
    ada = GridAda(*common_grid_params)

    estimators = [
        ('rf', rf),
        ('xgb', xgb),
        ('ada', ada)
    ]

    print("Estimators Ready")
    print("Voting Classifier Grid search running ... ")

    param_grid = {
        # 'voting' : ['hard', 'soft'],
        'voting': ['soft']
    }

    vc = VotingClassifier(estimators=estimators, n_jobs=-1)

    best_params = GridBackBone(clf=vc, param_grid=param_grid, X_train=X_train, y_train=y_train, cv=cv)

    print(best_params)

    best_vc = VotingClassifier(
        estimators=estimators,
        voting=best_params['voting'],
        n_jobs=-1)

    print("done")

    return best_vc
