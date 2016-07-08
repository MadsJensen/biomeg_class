import mne
from mne.channels.layout import _merge_grad_data as grad_rms
import scipy.io as sio
import numpy as np

from my_settings import data_folder

std_info = np.load(data_folder + "std_info.npy").item()

subject = 1

planar = sio.loadmat(data_folder + "meg_data_%sa.mat" % subject)["planardat"]

events = mne.read_events(data_folder + "sub_%s-eve.fif" % subject)

info = mne.create_info(204, ch_types="grad", sfreq=125)
info["chs"] = std_info["chs"]
info["ch_names"] = std_info["ch_names"]

info["lowpass"] = 41
info["highpass"] = 0.1

raw = mne.io.RawArray(planar, info)

event_id = {"Anger/non-target": 1,
            "Disgust/non-target": 2,
            "Fear/non-target": 3,
            "Happiness/target": 4,
            "Neutrality/non-target": 5,
            "Sadness/non-target": 6,
            "Test": 10}

tmin, tmax = -0.2, 0.83
reject = {"grad": 4000e-13}  # T / m (gradiometers)

epochs_params = dict(events=events,
                     event_id=event_id,
                     tmin=tmin,
                     tmax=tmax,
                     reject=reject,
                     baseline=(None, 0),
                     preload=True)

epochs = mne.Epochs(raw, **epochs_params)
epochs.save(data_folder + "sub_%s-epo.fif" % subject)

rms_data = np.asarray([grad_rms(t) for t in epochs.get_data()])
