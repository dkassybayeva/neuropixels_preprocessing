#%%
from behavior_utils import load_df, convert_df, trim_df
from mat73 import loadmat
from data_objs import TwoAFC, Multiday_2AFC
import datetime
import scipy.io as sio
import glob
import numpy as np

from trace_utils import create_traces_np, trial_start_align
#%%TQ02
datapath = glob.glob("G://Neurodata/Processed/TQ02/20210*")
datapath = sorted(datapath)[5:]

dates = [datetime.datetime(2021, 5, 23), datetime.datetime(2021, 5, 24), datetime.datetime(2021, 5, 25), datetime.datetime(2021, 5, 26), datetime.datetime(2021, 5, 27),
         datetime.datetime(2021, 5, 28),  datetime.datetime(2021, 5, 29), datetime.datetime(2021, 5, 30), datetime.datetime(2021, 5, 31),datetime.datetime(2021, 6, 1),
         datetime.datetime(2021, 6, 2), datetime.datetime(2021, 6, 3)]

#only including until 6/1 because of the mysteries of the switch
stimulus = ['freq', 'freq', 'freq', 'freq', 'freq_nat', 'freq_nat', 'freq_nat', 'nat', 'nat', 'nat_nat', 'nat', 'nat']

#0 means first day with noisy trials
#definition: "-1": good performance, no noise (or rare)
#definition: "0": first day with noise
#definition: "-2": poor performance, no noise (<70%)
ratname = "TQ02"
session_number = [-1, -1, 0, 1, -2, -2, -1, 0, 1, -1, 0, 1]
#%%
ratname = "TQ03"
datapath = ["H:\\Neurodata\\TQ03\\20210620_142900", "H:\\Neurodata\\TQ03\\20210621_153540",  "H:\\Neurodata\\TQ03\\20210622_152835",  "H:\\Neurodata\\TQ03\\20210623_153844" ]
dates = [datetime.datetime(2021, 6, 20), datetime.datetime(2021, 6, 21), datetime.datetime(2021, 6, 22), datetime.datetime(2021, 6, 23)]

#only including until 6/1 because of the mysteries of the switch
stimulus = ['clicks_nat', 'nat', 'nat', 'nat', 'nat']

#0 means first day with noisy trials
session_number = [-2, -1, -1, 0]

#%%
#TQ02\20210603_164551

data_objs = []
for i, dp in enumerate(datapath):

    #warnign magic number that obnly works on my file path

    #take the first cellbase folder
    cellbase = sorted(glob.glob(dp + '/cellbase*'))[-1]

    date = dates[i]#, datetime.datetime(2020, 12, 24), datetime.datetime(2021, 5, 13), datetime.datetime(2021, 6, 18)]


    timestep_ds = 25
    sps = 1000/timestep_ds

    name = ratname
    datpath = dp + '/'

    #make pandas behavior dataframe

    metadata = {'stimulus': stimulus[i],
                'time_investment':True,
                'reward_bias':False,
                'prior': False,
                'date': date,
                'experimenter': 'Amy',
                'region': 'lOFC',
                'behavior_phase': session_number[i],
                'recording_type': 'neuropixels',
                'experiment_id': 'learning_uncertainty',
                'linking_group': 'TQ02'}

    #load neural data
    data = loadmat(cellbase + "/traces_ms.mat")["spikes"]


    behav_df = load_df(datpath+"RecBehav.mat")

    if (i == 9) and (name == "TQ02"):
        behav_df['before_switch']  = np.zeros_like(behav_df.TrialNumber.to_numpy())
        behav_df.loc[0:250, 'before_switch'] = 1



    cbehav_df = convert_df(behav_df, type = "SessionData", WTThresh= 1, trim = True)

    if (i == 1) and (name == "TQ03"):
        cbehav_df = cbehav_df.iloc[:390]

    sps = 1000 / timestep_ds

    data, _ = trial_start_align(cbehav_df, data, 1000)

    data_ds = data.reshape(data.shape[0], data.shape[1], -1, timestep_ds).sum(axis = -1)


    #create trace alignments
    traces_dict = create_traces_np(cbehav_df, data_ds, sps = sps, aligned_ind = 0, filter_by_trial_num= False,
                                              traces_aligned = "TrialStart")
    cbehav_df['session'] = i
    cbehav_df = trim_df(cbehav_df)

    #create and save data object
    data_obj = TwoAFC(datpath, cbehav_df, traces_dict, name = name, cluster_labels = [],
                             metadata=metadata, sps = (1000/timestep_ds), record = True, feature_df_cache = [],
                                feature_df_keys = [])

    data_obj.to_pickle(remove_old = True)

    data_objs.append(data_obj)

#%%

import pandas as pd
matches = pickle.load(open(datapath + 'xday24_2.pickle', 'rb'))

behav_df = pd.concat([obj.behav_df for obj in obj_list])

#mobj is the concatenated traces and behav_df
#mobj = Multiday_2AFC(datapath, obj_list, matches, sps = sps, name='m_dan1_all', record = False)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import plot_condition_average

plt.subplot(1, 4, 1)
plt.plot(data_obj["correct", [0], 'stimulus'].mean(axis=0).T)

plt.subplot(1, 4, 2)
plt.plot(data_obj[:, [0], 'response'].mean(axis=0).T)

plt.subplot(1, 4, 3)
plt.plot(data_obj[{'stim_dir': 1}, [0], 'interp'].mean(axis=0).T)

ax = plt.subplot(1, 4, 4)
df = data_obj.get_feature_df(alignment='stimulus', variables=['stim_dir'])
plot_condition_average(df, variables=['stim_dir'], markers=[data_obj.stim_ind], ax=ax)
plt.tight_layout()
sns.despine()
plt.show()

#%%
