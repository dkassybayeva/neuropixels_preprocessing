"""
Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""
#%% Import native and custom libraries
import datetime
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mat73 import loadmat
from joblib import load

import neuropixels_preprocessing.lib.data_objs as data_objs


#%% Define data path and session variables

ratname = 'Nina2'
date = '20210623_121426'
probe_num = '2'
DATA_DIR = f"X:\\Neurodata\\{ratname}\\"
DATA_DIR += f"{date}.rec\\{date}.kilosort_probe{probe_num}"
path_list = glob.glob(DATA_DIR)
dates = [datetime.datetime(2021, 6, 23), ]

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

timestep_ds = 25  # sample period in ms
sps = 1000 / timestep_ds  # samples per second (1000ms)

metadata = {'time_investment': True,
            'reward_bias': False,
            'prior': False,  # Could possibly be Amy's code for a task type that was previously used
            'experimenter': 'Amy',
            'region': 'lOFC',
            'recording_type': 'neuropixels',
            'experiment_id': 'learning_uncertainty',
            'linking_group': 'Nina2'}


#%% Create data object for each experiment in the path_list
for i, dp in enumerate(path_list):
    data_objs.create_experiment_data_object(cellbase_dir, metadata, session_number=recording_session_id, sps=sps)

#%%

# from data_objs import Multiday_2AFC

# matches = pickle.load(open(datapath + 'xday24_2.pickle', 'rb'))

# behav_df = pd.concat([obj.behav_df for obj in obj_list])

# mobj is the concatenated traces and behav_df
# mobj = Multiday_2AFC(datapath, obj_list, matches, sps = sps, name='m_dan1_all', record = False)
