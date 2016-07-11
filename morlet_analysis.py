import numpy as np
import mne
from mne.time_frequency import cwt_morlet

import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

from my_settings import *


def combine_grad_tfr(tfr):
    """Merge tfr from channel pairs using the RMS

    Parameters
    ----------
    tfr : array, shape = (n_channels, n_times)
        tfr for channels, ordered in pairs.

    Returns
    -------
    tfr : array, shape = (n_channels / 2, n_times)
        The root mean square for each pair.
    """
    result = np.empty([tfr.shape[0], tfr.shape[1] / 2., tfr.shape[2],
                       tfr.shape[3]])

    for j in range(len(tfr)):
        tmp = tfr[j].reshape((len(tfr[j]) // 2, 2, -1))
        result[j, :, :, :] = np.reshape(
            np.sqrt(np.sum(tmp**2, axis=1) / 2),
            [tfr.shape[1] / 2., tfr.shape[2], tfr.shape[3]])

    return result


subject = 1

epochs = mne.read_epochs(data_folder + "sub_%s-epo.fif" % subject)

freqs = np.arange(4, 90, 3)
n_cycles = freqs / 3.

data_target = epochs["Happiness"].get_data()
data_nontarget = epochs["non-target"].get_data()

tfr_target = []
tfr_nontarget = []

for j in range(len(data_target)):
    tfr_target.append(cwt_morlet(data_target[j, :, :],
                                 sfreq=125,
                                 freqs=freqs,
                                 n_cycles=n_cycles))

for j in range(len(data_nontarget)):
    tfr_nontarget.append(cwt_morlet(data_nontarget[j, :, :],
                                    sfreq=125,
                                    freqs=freqs,
                                    n_cycles=n_cycles))

# Convert to numpy arrays
tfr_target = np.asarray(tfr_target)
tfr_nontarget = np.asarray(tfr_nontarget)

# Take power of signal
pow_target = np.abs(tfr_target)**2
pow_nontarget = np.abs(tfr_nontarget)**2

comb_target = combine_grad_tfr(pow_target)
comb_nontarget = combine_grad_tfr(pow_nontarget)

times = epochs.times
comb_target_bs = mne.baseline.rescale(comb_target,
                                      times,
                                      baseline=(None, 0),
                                      mode="zscore")

comb_nontarget_bs = mne.baseline.rescale(comb_nontarget,
                                         times,
                                         baseline=(None, 0),
                                         mode="zscore")

# classification
X = np.concatenate([comb_target_bs.reshape([len(comb_target_bs), -1]),
                    comb_nontarget_bs.reshape([len(comb_nontarget_bs), -1])])
y = np.concatenate(
    [np.zeros(len(comb_target_bs)), np.ones(len(comb_nontarget_bs))])

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

scores_xgb = cross_val_score(ada_cv, X, y, cv=cv)
print(scores_xgb)
joblib.dump(grid, class_data + "sub_%s-xgb_grid.pkl" % subject)

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

scores_ada = cross_val_score(ada_cv, X, y, cv=cv)
print(scores_ada)
joblib.dump(grid_ada, class_data + "sub_%s-ada_grid.pkl" % subject)
