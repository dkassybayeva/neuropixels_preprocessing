# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""
# %% 
from joblib import load, dump

import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.behavior_utils as bu
import neuropixels_preprocessing.lib.data_objs as data_objs
from neuropixels_preprocessing.session_params import *
import neuropixels_preprocessing.lib.trace_utils as trace_utils


#%%
#----------------------------------------------------------------------#
# The information in the metadata block of session_params needs to be
# filled out and updated for each recording session.
#----------------------------------------------------------------------#
SPIKES_AND_TTL = True
SAVE_INDIVIDUAL_SPIKETRAINS = True
BEHAVIOR = True


metadata = dict(
    ott_lab = True,
    rat_name = 'R12',
    date = '20231210',
    behavior_mat_file = '12_DetectionConfidence_20231210_183107.mat',
    trodes_datetime = '20231210_191835',
    n_probes = 2,
    DIO_port_num = 1,
    task = 'reward-bias',
    sps = sps,
    task_type = 'DetectionConfidence',
    probe_num = 2,
    kilosort_ver = 4
)
#----------------------------------------------------------------------#

# %%
#----------------------------------------------------------------------#
#                       Set up paths
#----------------------------------------------------------------------#
session_paths = dict()
session_paths['rec_dir'] = rec_dir = f'X:{metadata["rat_name"]}/{metadata["trodes_datetime"]}.rec/'
assert path.exists(session_paths['rec_dir'])
# session path for ks4
session_paths['probe_dir'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.kilosort/{metadata["trodes_datetime"]}.kilosort{metadata["kilosort_ver"]}'+'_probe{}/'
# session path for ks2.5
#session_paths['probe_dir'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.kilosort{metadata["kilosort_ver"]}'+'_probe{}/'

session_paths['timestamps_dat'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.kilosort/{metadata["trodes_datetime"]}.timestamps.dat'

session_paths['preprocess_dir'] = preprocess_dir =  session_paths['rec_dir'] + 'preprocessing_output/'
ou.make_dir_if_nonexistent(preprocess_dir)

behavior_mat_file = session_paths['rec_dir'] + metadata['behavior_mat_file']
#----------------------------------------------------------------------#

# %%
#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if SPIKES_AND_TTL:
    for probe_i in range(1, metadata['n_probes']+1):
        metadata['probe_num'] = probe_i

        # load/create ephys-specific (probe-specific) paths
        probe_dir = session_paths['probe_dir'].format(probe_i)
        spike_dir = preprocess_dir + f'probe{probe_i}/'
        ou.make_dir_if_nonexistent(spike_dir)
        if SAVE_INDIVIDUAL_SPIKETRAINS:
            ou.make_dir_if_nonexistent(spike_dir + 'spike_times/')

        # process ephys recordings
        tu.create_spike_mat(probe_dir, spike_dir, session_paths['timestamps_dat'], metadata, fs, save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)

    tu.find_recording_gaps(session_paths['timestamps_dat'], fs, max_ISI, preprocess_dir + gap_filename)
    tu.extract_TTL_trial_start_times(probe_dir, gap_filename, metadata['DIO_port_num'], save_dir=preprocess_dir)
    tu.reconcile_TTL_and_behav_trial_start_times(rec_dir, preprocess_dir, behavior_mat_file)
    
  # %% Cut Recording into Auditory Tuning and Detection Confidence based on TTL
  
  
# %%
# if BEHAVIOR:
#     _sd = load(preprocess_dir + 'TrialEvents.npy')
    

#     n_trials = _sd['nTrials'] - 1  # throw out last trial (may be incomplete)
    
    
#     trialstart_str = 'recorded_TTL_trial_start_time'
#     trial_len = _sd[trialstart_str][1:] - _sd[trialstart_str][:-1]
    
#     behav_dict = dict(
#     volume = _sd['Custom']['Volume'][:n_trials],
#     frequency = _sd['Custom']['Frequency'][:n_trials],
#     TrialStartTimestamp = _sd['TrialStartTimestamp'][:n_trials],
#     TrialEndTimestamp = _sd['TrialEndTimestamp'][:n_trials],
#     recorded_TTL_trial_start_time = _sd[trialstart_str][:n_trials],
#     trial_len = trial_len[:n_trials]
#     )
#     behav_df = pd.DataFrame.from_dict(behav_dict)
    
#     dump(behav_df, preprocess_dir + "behav_df", compress=3)

# downsample_dt = 25  # sample period in ms
# # %%
# #----------------------------------------------------------------------#
# # Align the spikes with behavior
# for probe_i in range(1, metadata['n_probes']+1):
#     metadata['probe_num'] = probe_i
    
#     # load neural data: [number of neurons x time bins in ms]
#     spike_mat = load(preprocess_dir + f"probe{probe_i}/" + spike_mat_str_indiv)['spike_mat']
#     behav_df = load(preprocess_dir + 'behav_df')
    
    
#     # align spike times to behavioral data timeframe
#     # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
#     trialwise_spike_mat_start_aligned, _ = trace_utils.trial_start_align(behav_df, spike_mat, metadata, sps=1000)
    
#     # subsample (bin) data:
#     # [n_neurons x n_trials x (-1 means numpy calculates: trial_len / dt) x ds]
#     # then sum over the dt bins
#     n_neurons = trialwise_spike_mat_start_aligned.shape[0]
#     n_trials = trialwise_spike_mat_start_aligned.shape[1]
#     trial_binned_mat_start_align = trialwise_spike_mat_start_aligned.reshape(n_neurons, n_trials, -1, downsample_dt)
#     trial_binned_mat_start_align = trial_binned_mat_start_align.sum(axis=-1)  # sum over bins
    
#     results = {'binned_mat': trial_binned_mat_start_align, 'downsample_dt': downsample_dt}
#     dump(results, preprocess_dir + f"probe{metadata['probe_num']}/trial_binned_mat_start_align.npy", compress=3)
