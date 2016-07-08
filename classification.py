import xgboost as xgb
import numpy as np
import mne

from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

from my_settings import *

subject = 1

epochs = mne.read_epochs(data_folder + "sub_%s-epo.fif" % subject)

data_shape = epochs.get_data().shape

X = epochs[:239].get_data().reshape([239, data_shape[1] * data_shape[2]])
y = (epochs[:239].events[:, 2] != 4).astype("int")

cv = StratifiedShuffleSplit(y, test_size=0.1)

cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.2),
             "max_depth": [1, 3, 5, 7],
             "n_estimators": np.arange(100, 1100, 100)}

grid = GridSearchCV(xgb.XGBClassifier(),
                    cv_params,
                    scoring='roc_auc',
                    cv=cv,
                    n_jobs=-1,
                    verbose=1)
grid.fit(X, y)
xgb_cv = grid.best_estimator_

joblib.dump(xgb_cv, class_data + "sub_%s-xgb_model.pkl" % subject)

cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.1),
             'n_estimators': np.arange(1, 2000, 200)}

grid_ada = GridSearchCV(AdaBoostClassifier(),
                        cv_params,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=-1,
                        verbose=1)
grid_ada.fit(X, y)
ada_cv = grid_ada.best_estimator_

scores = cross_val_score(ada_cv, X, y, cv=cv)

joblib.dump(ada_cv, class_data + "sub_%s-ada_model.pkl" % subject)
