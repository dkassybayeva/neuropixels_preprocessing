from os.path import exists
import numpy as np
from scipy.io.matlab import loadmat
import joblib
import seaborn as sns

from neuropixels_preprocessing.session_params import get_root_path 
from neuropixels_preprocessing.lib.behavior_utils import calc_event_outcomes
import neuropixels_preprocessing.lib.trace_utils as tu

# -------------------------------------------------------------- #
#                   File name and path
# -------------------------------------------------------------- #
DATA_ROOT = 'server'  # ['local', 'server', 'X:', etc.]
rat = 'Nina2'
date = '20210622'
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
sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces
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

"""
The following is equivalent to:
spike_mat = spike_mat.reshape(T,  n_trials, n_neurons)
spike_mat = spike_mat.transpose(2, 1, 0)
Matlab uses Fortran (column) matrix reordering, whereas numpy defaults to 'C' (row) reordering.
"""
spike_mat = spike_mat.reshape(n_neurons, n_trials, T, order='F')

assert spike_mat.sum() == len(spike_times)

assert n_trials == len(event_dict['TrialStartAligned'])
behav_df = calc_event_outcomes(event_dict, metadata)

unrewarded_idx = (event_dict['Rewarded']==0) & ~np.isnan(event_dict['ChosenDirection'])
heatmap = np.nanmean(spike_mat[:, unrewarded_idx, :], 1)
hmax = heatmap.max().round()
sns.heatmap(heatmap, cmap='vlag', cbar=True, vmin=-hmax, vmax=hmax)
# -------------------------------------------------------------- #


# -------------------------------------------------------------- #
#                   Filter choice trials
# -------------------------------------------------------------- #
choice_mask = behav_df[behav_df['MadeChoice']].index.to_numpy()
cbehav_df = behav_df[behav_df['MadeChoice']].reset_index(drop=True)
joblib.dump(cbehav_df, DATA_DIR + "behav_df", compress=3)
# -------------------------------------------------------------- #


# -------------------------------------------------------------- #
#                   Create aligned traces
# -------------------------------------------------------------- #
trialwise_binned_response_align = spike_mat[:, choice_mask, :]
assert trialwise_binned_response_align.shape[1] == len(cbehav_df)

"""
For testing: this is a good breakpoint, where the data can be loaded instead of recalculated
every time a new alignment or interpolation is tried.
"""
# cbehav_df = joblib.load(DATA_DIR + 'behav_df')
# trialwise_binned_response_align = joblib.load(DATA_DIR + 'trialwise_binned_response_align_choice_only_ms')

alignment_param_dict = dict(
    trial_times_in_reference_to='ResponseStart',  # ['TrialStart', 'ResponseStart']
    resp_start_align_buffer=int(2.0 * sps),  # for ResponseStart
    downsample_dt=trace_subsample_bin_size_ms,
    pre_stim_interval = int(0.5 * sps),  # truncated at center_poke
    post_stim_interval = int(0.5*sps),  # truncated at stim_off
    pre_response_interval = int(3.0*sps),  # truncated at stim_off
    post_response_interval = int(4.0*sps),  # truncated at response_end
    pre_reward_interval = int(6.0*sps),  # truncated at response_time
    post_reward_interval = int(5.0*sps),  # truncated at trial_end
)

tu.align_traces_to_task_events(cbehav_df, trialwise_binned_response_align, alignment_param_dict, save_dir=DATA_DIR)


# -------------------------------------------------------------- #
#                   Create interpolated traces
# -------------------------------------------------------------- #
interpolation_param_dict = dict(
    trial_times_in_reference_to='ResponseStart',  # ['TrialStart', 'ResponseStart']
    resp_start_align_buffer=int(2.0 * sps),  # for ResponseStart
    trial_event_interpolation_lengths = [
        int(0.5 * sps),  # ITI->center poke
        int(0.45 * sps), # center->stim_begin
        int(.5 * sps),   # stim delivery
        int(.3 * sps),   # movement to side port
        # int(0.5 * sps),  # first 0.5s of anticipation epoch
        # int(0.5 * sps),  # second part of anticipation epoch warped into 0.5s (actually half second in reward-bias)
        int(3.0 * sps),  # anticipation epoch
        int(1.5 * sps),  # after feedback
    ],
    pre_center_interval = int(0.5 * sps),
    post_response_interval = None,  # int(0.5 * sps) or None.  If None, then the midpoint between response start and end is used
    downsample_dt=trace_subsample_bin_size_ms,
)

tu.interpolate_traces(cbehav_df, trialwise_binned_response_align, interpolation_param_dict, save_dir=DATA_DIR)

