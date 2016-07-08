import xgboost as xgb
import numpy as np
import mne

from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from my_settings import *

subject = 1

epochs = mne.read_epochs(data_folder + "sub_%s-epo.fif" % subject)

X = epochs[:239].get_data().reshape([239, 26928])
y = epochs[:239].events[:, 2]
y = (y != 4)
y = y.astype("int")

cv = StratifiedShuffleSplit(y, test_size=0.1)

cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.2),
             "max_depth": [1, 3, 5, 7]}

grid = GridSearchCV(xgb.XGBClassifier(n_estimators=500),
                    cv_params,
                    scoring='roc_auc',
                    cv=cv,
                    n_jobs=3,
                    verbose=2)
grid.fit(X, y)
xgb_cv = grid.best_estimator_


cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.1),
             'n_estimators': np.arange(1, 2000, 200)}

grid_ada = GridSearchCV(AdaBoostClassifier(),
                        cv_params,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=3,
                        verbose=2)
grid_ada.fit(X, y)
ada_cv = grid_ada.best_estimator_

scores = cross_val_score(ada_cv, X, y, cv=cv)
