#%% Import native and custom libraries
import datetime
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mat73 import loadmat


# ---------- Amy Functions ------------ #
from behavior_utils import load_df, convert_df, trim_df
from data_objs import TwoAFC
from trace_utils import create_traces_np, trial_start_align

#%% Define data path and session variables

ratname = 'Nina2'
date = '20210623_121426'
probe_num = '2'
full_data_dir = f"X:\\Neurodata\\{ratname}\\"
full_data_dir += f"{date}.rec\\{date}.kilosort_probe{probe_num}"
path_list = glob.glob(full_data_dir)
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
    
    # load neural data: [number of neurons x time bins in ms]
    spike_times = loadmat(cellbase + "traces_ms.mat")["spikes"]
    
    # make pandas behavior dataframe
    behav_df = load_df(cellbase + "RecBehav.mat")

    # format entries of dataframe for analysis (e.g., int->bool)
    cbehav_df = convert_df(behav_df, session_type="SessionData", WTThresh=1, trim=True)
    
    # align spike times to behavioral data
    # spike_times = array [n_neurons x n_trials x longest_trial period in ms]
    spike_times, _ = trial_start_align(cbehav_df, spike_times, 1000)
    
    # subsample (bin) data:
    # [n_neurons x n_trials x (-1 means numpy calculates: trial_len / dt) x ds]
    # then sum over the dt bins
    n_neurons = spike_times.shape[0]
    n_trials = spike_times.shape[1]
    spike_times_ds = spike_times.reshape(n_neurons, n_trials, -1, timestep_ds)
    spike_times_ds = spike_times_ds.sum(axis=-1)
    
    # create trace alignments
    traces_dict = create_traces_np(cbehav_df, 
                                   spike_times_ds,
                                   sps=sps, 
                                   aligned_ind=0, 
                                   filter_by_trial_num=False,
                                   traces_aligned="TrialStart")
    
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
    data_obj_i = create_experiment_data_object(i, dp)
    data_list.append(data_obj_i)

#%%

# from data_objs import Multiday_2AFC

# matches = pickle.load(open(datapath + 'xday24_2.pickle', 'rb'))

# behav_df = pd.concat([obj.behav_df for obj in obj_list])

# mobj is the concatenated traces and behav_df
# mobj = Multiday_2AFC(datapath, obj_list, matches, sps = sps, name='m_dan1_all', record = False)
