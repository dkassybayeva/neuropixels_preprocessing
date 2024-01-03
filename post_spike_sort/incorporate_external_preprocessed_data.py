import numpy as np
from scipy.io.matlab import loadmat

from neuropixels_preprocessing.lib.behavior_utils import calc_event_outcomes
from neuropixels_preprocessing.lib.data_objs import create_experiment_data_object
from neuropixels_preprocessing.lib.obj_utils import make_dir_if_nonexistent

rat_date = 'Nina2_210623'
input_data = rat_date + 'a_ms_PSTH.mat'
trace_subsample_bin_size_ms = 10  # sample period in ms

metadata = dict(
    ott_lab=False,
    task='time-investment',
    sps = 1000 / trace_subsample_bin_size_ms,  # (samples per second) resolution of aligned traces
)

DATA_DIR = '/home/mud/Workspace/ott_neuropix_data/Neuropixels_dataset/'
OUTPUT_DIR = DATA_DIR+rat_date+'/'
make_dir_if_nonexistent(OUTPUT_DIR)

session_data = loadmat(DATA_DIR+input_data, simplify_cells=True)

event_dict = session_data['TE']
n_neurons, n_trials, T = session_data['PopulationPSTH_dims']
spike_times = session_data['SPIKEIDX']

spike_mat = np.zeros(int(n_neurons) * int(n_trials) * int(T), dtype='uint8')
spike_mat[spike_times.astype(int)] = 1
spike_mat = spike_mat.reshape(n_neurons, n_trials, T)
assert spike_mat.sum() == len(spike_times)
assert n_trials == len(event_dict['TrialStartAligned'])

trialwise_binned_start_align = spike_mat.reshape(n_neurons, n_trials, -1, trace_subsample_bin_size_ms)
trialwise_binned_start_align = trialwise_binned_start_align.sum(axis=-1)  # sum over bins


behav_df = calc_event_outcomes(event_dict, metadata)
choice_mask = behav_df[behav_df['MadeChoice']].index.to_numpy()
cbehav_df = behav_df[behav_df['MadeChoice']].reset_index(drop=True)
trialwise_binned_start_align = trialwise_binned_start_align[:, choice_mask, :]
assert trialwise_binned_start_align.shape[1] == len(cbehav_df)

create_experiment_data_object(data_path=OUTPUT_DIR,
                              metadata=metadata,
                              nrn_phy_ids=[],
                              trialwise_binned_mat=trialwise_binned_start_align,
                              cbehav_df=cbehav_df)