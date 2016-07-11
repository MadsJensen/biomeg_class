# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:40:32 2016

@author: au194693
"""

import mne
import numpy as np

from my_settings import *

subject = 1

raw = mne.io.Raw(data_folder + "sub_%s-raw.fif" % subject, preload=True)
raw.filter(8, 12)

picks = mne.pick_types(raw.info, "grad")
raw.apply_hilbert(picks)

events = mne.read_events(data_folder + "sub_%s-eve.fif" % subject)
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

ave_nt = epochs["non-target"].average()
data = np.abs(epochs["non-target"].get_data())**2
ave_nt.data = data.mean(axis=0)

data = np.abs(epochs["Happiness"].get_data())**2
ave_happines = epochs["Happiness"].average()
ave_happines.data = data.mean(axis=0)

data = np.abs(epochs["Anger"].get_data())**2
ave_anger = epochs["Anger"].average()
ave_anger.data = data.mean(axis=0)

data = np.abs(epochs["Fear"].get_data())**2
ave_fear = epochs["Fear"].average()
ave_fear.data = data.mean(axis=0)

data = np.abs(epochs["Disgust"].get_data())**2
ave_disgust = epochs["Disgust"].average()
ave_disgust.data = data.mean(axis=0)

data = np.abs(epochs["Sadness"].get_data())**2
ave_sadness = epochs["Sadness"].average()
ave_sadness.data = data.mean(axis=0)

data = np.abs(epochs["Neutrality"].get_data())**2
ave_neutral = epochs["Neutrality"].average()
ave_neutral.data = data.mean(axis=0)

evokeds = [ave_happines, ave_anger, ave_disgust, ave_fear, ave_neutral,
           ave_sadness]
colors = ["red", "green", "lightblue", "m", "orange", "yellow"]

result = np.empty([data.shape[1], data.shape[2]])

for j in range(data.shape[1]):
    for i in range(data.shape[2]):
        result[j, i] =\
            np.abs(np.mean(np.exp(1j * (np.angle(data[:, j, i])))))
