import mne
import scipy.io as sio

from my_settings import data_folder

planar = sio.loadmat(data_folder + "meg_data_1a.mat")["planardat"]

events = mne.read_events(data_folder + "sub_1-eve.fif")

info = mne.create_info(204, ch_types="grad", sfreq=125)

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

tmin, tmax = -0.25, 0.8

reject = {"grad": 4000e-13}  # T / m (gradiometers)

epochs_params = dict(events=events,
                     event_id=event_id,
                     tmin=tmin,
                     tmax=tmax,
                     reject=reject,
                     baseline=(None, 0),
                     preload=True)

epochs = mne.Epochs(raw, **epochs_params)
