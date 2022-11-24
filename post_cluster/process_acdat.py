#%% Import native and custom libraries
import datetime
# import scipy.io as sio
import glob
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Amy Functions ------------ #
from behavior_utils import load_df, convert_df, trim_df
from mat73 import loadmat
from data_objs import TwoAFC #, Multiday_2AFC
from plotting_utils import plot_condition_average
from trace_utils import create_traces_np, trial_start_align

#%% Define data path and session variables

ratname = 'Nina2'
date = '20210623_121426'
probe_num = '2'
full_data_dir = f"X:\\Neurodata\\{ratname}\\"
full_data_dir += f"{date}.rec\\{date}.kilosort_probe{probe_num}"
path_list = glob.glob(full_data_dir)
dates = [datetime.datetime(2021, 6, 23),]

"""
List of stimuli for each experiment:
    'freq' = 
    'freq_nat' = 
    'nat' = 
    'nat_nat' =
"""
stimulus = ['freq']

"""
Session number code:
    -1 = good performance, no noise (or rare)
     0 = first day with noise
    -2 = poor performance, no noise (<70%)
"""
session_number = [-1]

#%% Parameters and Metadata

timestep_ds = 25
sps = 1000 / timestep_ds

metadata = {'time_investment':True,
            'reward_bias':False,
            'prior': False,
            'experimenter': 'Amy',
            'region': 'lOFC',
            'recording_type': 'neuropixels',
            'experiment_id': 'learning_uncertainty',
            'linking_group': 'Nina2'}

#%% Define data function

def create_experiment_data_object(i, datapath):
    cellbase = datapath + '/cellbase/'
    
    meta_d = metadata.copy()
    meta_d['date'] = dates[i]
    meta_d['stimulus'] = stimulus[i]
    meta_d['behavior_phase'] = session_number[i]
    

    # load neural data
    data = loadmat(cellbase + "traces_ms.mat")["spikes"]
    
    # make pandas behavior dataframe
    behav_df = load_df(cellbase + "RecBehav.mat")

    cbehav_df = convert_df(behav_df, session_type="SessionData", WTThresh=1, trim=True)
    
    # align and reshape data
    data, _ = trial_start_align(cbehav_df, data, 1000)
    
    data_ds = data.reshape(data.shape[0], data.shape[1], -1, timestep_ds).sum(axis = -1)
    
    
    # create trace alignments
    traces_dict = create_traces_np(cbehav_df, 
                                   data_ds, 
                                   sps=sps, 
                                   aligned_ind=0, 
                                   filter_by_trial_num= False, 
                                   traces_aligned = "TrialStart")
    
    cbehav_df['session'] = i
    cbehav_df = trim_df(cbehav_df)
    
    # create and save data object
    data_obj = TwoAFC(datapath, cbehav_df, traces_dict, name=ratname, 
                      cluster_labels=[], metadata=meta_d, sps=sps, 
                      record=False, feature_df_cache=[], feature_df_keys=[])
    
    data_obj.to_pickle(remove_old=False)
    
    return data_obj


#%% Create data object for each experiment in the path_list

data_list = []
for i, dp in enumerate(path_list):
    data_obj = create_experiment_data_object(i, dp)
    data_list.append(data_obj)

#%%


# matches = pickle.load(open(datapath + 'xday24_2.pickle', 'rb'))

# behav_df = pd.concat([obj.behav_df for obj in obj_list])

# mobj is the concatenated traces and behav_df
#mobj = Multiday_2AFC(datapath, obj_list, matches, sps = sps, name='m_dan1_all', record = False)

#%% Plot results for first data object

data_obj = data_list[0]

nrn_resp = data_obj["correct", [0], 'stimulus'].mean(axis=0).T
ax1 = plt.subplot(1, 4, 1)
plt.plot(nrn_resp)
plt.xlim(0, nrn_resp.shape[0])
plt.ylim(0, 0.3)
plt.yticks([0.1, 0.2, 0.3])
ax1.spines['top'].set_visible(0)
ax1.spines['right'].set_visible(0)

nrn_resp = data_obj[:, [0], 'response'].mean(axis=0).T
ax2 = plt.subplot(1, 4, 2)
plt.plot(nrn_resp)
plt.xlim(0, nrn_resp.shape[0])
plt.ylim(0, 0.3)
plt.yticks([0.1, 0.2, 0.3])
ax2.spines['top'].set_visible(0)
ax2.spines['right'].set_visible(0)

nrn_resp = data_obj[{'stim_dir': 1}, [0], 'interp'].mean(axis=0).T
ax3 = plt.subplot(1, 4, 3)
plt.plot(nrn_resp)
plt.xlim(0, nrn_resp.shape[0])
plt.ylim(0, 0.5)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax3.spines['top'].set_visible(0)
ax3.spines['right'].set_visible(0)

ax = plt.subplot(1, 4, 4)
df = data_obj.get_feature_df(alignment='stimulus', variables=['stim_dir'])
plot_condition_average(df, variables=['stim_dir'], markers=[data_obj.stim_ind], ax=ax)

plt.tight_layout()
sns.despine()
plt.show()

