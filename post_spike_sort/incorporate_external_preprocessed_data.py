from os.path import exists
import numpy as np
from scipy.io.matlab import loadmat
import joblib

from session_params import get_root_path 
from neuropixels_preprocessing.lib.behavior_utils import calc_event_outcomes
from neuropixels_preprocessing.lib.trace_utils import create_traces_np

# -------------------------------------------------------------- #
#                   File name and path
# -------------------------------------------------------------- #
DATA_ROOT = 'server'  # ['local', 'server', 'X:', etc.]
rat = 'Nina2'
date = '20210623'
input_data = f'{rat}_{date}a_ms_PSTH.mat'

DATA_DIR = get_root_path(DATA_ROOT) + f'{rat}/ephys/{date}_Torben_preprocess/'
assert exists(DATA_DIR + input_data)
metadata = dict(
    ott_lab=False,
    rat_name = rat,
    date = date,
    n_probes = 2,
    task='time-investment',
)

# -------------------------------------------------------------- #

# -------------------------------------------------------------- #
#                  Preprocessing parameters
# -------------------------------------------------------------- #
trace_subsample_bin_size_ms = 10  # sample period in ms
sps = 1000 / trace_subsample_bin_size_ms,  # (samples per second) resolution of aligned traces
# -------------------------------------------------------------- #

# -------------------------------------------------------------- #
#            Load data into spike mat and behav_df
# -------------------------------------------------------------- #
session_data = loadmat(DATA_DIR+input_data, simplify_cells=True)

event_dict = session_data['TE']
n_neurons, n_trials, T = session_data['PopulationPSTH_dims']
spike_times = session_data['SPIKEIDX']

spike_mat = np.zeros(int(n_neurons) * int(n_trials) * int(T), dtype='uint8')
spike_mat[spike_times.astype('uint')] = 1
spike_mat = spike_mat.reshape(n_neurons, n_trials, T)
assert spike_mat.sum() == len(spike_times)

assert n_trials == len(event_dict['TrialStartAligned'])
behav_df = calc_event_outcomes(event_dict, metadata)
# -------------------------------------------------------------- #


# -------------------------------------------------------------- #
#                   Filter choice trials
# -------------------------------------------------------------- #
choice_mask = behav_df[behav_df['MadeChoice']].index.to_numpy()
cbehav_df = behav_df[behav_df['MadeChoice']].reset_index(drop=True)
joblib.dump(cbehav_df, DATA_DIR + "behav_df", compress=3)
# -------------------------------------------------------------- #


# -------------------------------------------------------------- #
#         Subsample spiking and create aligned traces
# -------------------------------------------------------------- #
trialwise_binned_start_align = spike_mat.reshape(n_neurons, n_trials, -1, trace_subsample_bin_size_ms)
trialwise_binned_start_align = trialwise_binned_start_align.sum(axis=-1)  # sum over bins

trialwise_binned_start_align = trialwise_binned_start_align[:, choice_mask, :]
assert trialwise_binned_start_align.shape[1] == len(cbehav_df)

_ = create_traces_np(cbehav_df, trialwise_binned_start_align, trace_subsample_bin_size_ms, DATA_DIR,
                                 aligned_ind=0, filter_by_trial_num=False, traces_aligned="TrialStart")
