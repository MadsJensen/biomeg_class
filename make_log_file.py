import pandas as pd
import numpy as np
import mne

subject = 1

from my_settings import *

triggers = pd.melt(
    pd.read_csv(data_folder + 'triggers_sub_%s.csv' % subject),
    value_name='sample',
    var_name="trigger")
test_triggers = pd.read_csv(data_folder + "test_triggers_sub_%s.csv" % subject, header=None)
test_triggers.columns = ["test"]

test_triggers = pd.melt(test_triggers, value_name="sample", var_name="trigger")

all_triggers = pd.concat([triggers, test_triggers]).reset_index()
all_triggers["empty"] = 0

int_trigs = np.concatenate([np.ones(40), np.ones(40) * 2, np.ones(40) * 3,
                            np.ones(40) * 4, np.ones(40) * 5, np.ones(40) * 6,
                            np.ones(240) * 10])

all_triggers["int_trig"] = int_trigs
triggers = all_triggers[["sample", "empty", "int_trig"]].sort_values(by="sample").get_values()

mne.write_events(data_folder + "sub_%s-eve.fif" % subject, triggers)
