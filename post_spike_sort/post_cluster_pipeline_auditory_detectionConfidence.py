# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:33:50 2024

@author: Dariya Kassybayeva
"""
# %% 
import os
import numpy as np
from tqdm import trange
from joblib import load, dump

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile, get_Trodes_timestamps
import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.behavior_utils as bu
import neuropixels_preprocessing.lib.data_objs as data_objs
import neuropixels_preprocessing.lib.trace_utils as trace_utils
from neuropixels_preprocessing.session_params import *
import pandas as pd

#%%
#----------------------------------------------------------------------#
# The information in the metadata block of session_params needs to be
# filled out and updated for each recording session.
#----------------------------------------------------------------------#
DATA_ROOT = 'Y:'  # ['local', 'server', 'X:', etc.]
SPIKES_AND_TTL = False
BEHAVIOR = True
LFPs = False
DATA_OBJECT = True

SAVE_INDIVIDUAL_SPIKETRAINS = True
WRITE_METADATA = True
    
if WRITE_METADATA:
    metadata = write_session_metadata_to_csv(DATA_ROOT)
else:
    rat = 'R13'
    date = '20231213'
    metadata = load_session_metadata_from_csv(DATA_ROOT, rat, date)
#%%   
# #%% old
# #----------------------------------------------------------------------#
# # The information in the metadata block of session_params needs to be
# # filled out and updated for each recording session.
# #----------------------------------------------------------------------#
# SPIKES_AND_TTL = False
# SAVE_INDIVIDUAL_SPIKETRAINS = False  # Used for stitching sessions.  Otherwise unnecessary.
# BEHAVIOR = True


# metadata = dict(
#     ott_lab = True,
#     rat_name = 'R13',
#     date = '20231213',
#     behavior_mat_file = ['13_AuditoryTuning_20231213_131543.mat', '13_DetectionConfidence_20231213_160823.mat'],
#     trodes_datetime = '20231213_155419',
#     n_probes = 1,
#     DIO_port_num = 1,
#     task = 'reward-bias',
#     sps = sps,
#     task_type = 'DetectionConfidence',
#     probe_num = 1,
#     kilosort_ver = 4
# )
# #%% 
# # ----------------------------------------------------------------------- #
# #                           Constants
# # ----------------------------------------------------------------------- #
# # -------recording--------- #
# fs = 30e3  # Trodes sampling frequency in Hz
# ms_converter = 1000 / fs
# n_Npix1_electrodes = 960
# n_active_electrodes = 384

# # -------analysis bins--------- #
# max_ISI = 0.001  # max intersample interval (ISI), above which the period is considered a "gap" in the recording
# trace_subsample_bin_size_ms = 10 # sample period in ms
# sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces

# # -------params for trace interpolation------- #
# """
# For the source of these numbers, see 'Temporal dynaics clustering' in Hirokawa et al. Nature (2019) in Methods.
# """
# interpolation_param_dict = dict(
#     trial_times_in_reference_to='TrialStart',  # ['TrialStart', 'ResponseStart']
#     resp_start_align_buffer=None,  # for ResponseStart
#     trial_event_interpolation_lengths = [
#         int(0.5 * sps),  # ITI->center poke
#         int(0.45 * sps), # center->stim_begin
#         int(.5 * sps),   # stim delivery
#         int(.3 * sps),   # movement to side port
#         # int(0.5 * sps),  # first 0.5s of anticipation epoch
#         # int(0.5 * sps),  # second part of anticipation epoch warped into 0.5s (actually half second in reward-bias)
#         int(3.0 * sps),  # anticipation epoch
#         int(1.5 * sps),  # after feedback
#     ],
#     pre_center_interval = int(0.5 * sps),
#     post_response_interval = None,  # int(0.5 * sps) or None.  If None, then the midpoint between response start and end is used
#     downsample_dt=trace_subsample_bin_size_ms,
# )

# alignment_param_dict = dict(
#     trial_times_in_reference_to='TrialStart',  # ['TrialStart', 'ResponseStart']
#     resp_start_align_buffer=None,  # for ResponseStart
#     downsample_dt=trace_subsample_bin_size_ms,
#     pre_stim_interval = int(0.5 * sps),  # truncated at center_poke
#     post_stim_interval = int(0.5*sps),  # truncated at stim_off
#     pre_response_interval = int(3.0*sps),  # truncated at stim_off
#     post_response_interval = int(4.0*sps),  # truncated at response_end
#     pre_reward_interval = int(6.0*sps),  # truncated at response_time
#     post_reward_interval = int(5.0*sps),  # truncated at trial_end
# )



# # ---------file names---------- #
# spike_mat_str_indiv = f'spike_mat_in_ms.npy'
# gap_filename = f"trodes_intersample_periods_longer_than_{max_ISI}s.npy"
# ---------------------------------------------------------------------


#----------------------------------------------------------------------#

# %%
#----------------------------------------------------------------------#
#                       Set up paths
#----------------------------------------------------------------------#
session_paths = dict()
session_paths['rec_dir'] = rec_dir = f'Y:{metadata["rat_name"]}/ephys/{metadata["trodes_datetime"]}.rec/'
assert path.exists(session_paths['rec_dir'])
# session path for spikeinterface with ks4
session_paths['probe_dir'] = session_paths['rec_dir'] + f'spike_interface_output/' + '{}/sorter_output/'
# session path for ks4
#session_paths['probe_dir'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.kilosort/{metadata["trodes_datetime"]}.kilosort{metadata["kilosort_ver"]}'+'_probe{}/'
# session path for ks2.5
#session_paths['probe_dir'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.kilosort{metadata["kilosort_ver"]}'+'_probe{}/'

session_paths['timestamps_dat'] = session_paths['rec_dir'] + f'{metadata["trodes_datetime"]}.timestamps.dat'

session_paths['preprocess_dir'] = preprocess_dir =  session_paths['rec_dir'] + 'preprocessing_output/'
ou.make_dir_if_nonexistent(preprocess_dir)

session_paths['preprocess_dir_auditory'] = preprocess_dir_auditory =  session_paths['rec_dir'] + 'preprocessing_output/auditoryTuning/'
ou.make_dir_if_nonexistent(preprocess_dir_auditory)

session_paths['preprocess_dir_dc'] = preprocess_dir_dc =  session_paths['rec_dir'] + 'preprocessing_output/detectionConfidence/'
ou.make_dir_if_nonexistent(preprocess_dir_dc)

behavior_mat_file = [session_paths['rec_dir'] + metadata['behavior_mat_file'][0], session_paths['rec_dir'] + metadata['behavior_mat_file'][1]]
#----------------------------------------------------------------------#

# %%
#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#

  # %% Align start times of TTL and spike_times.npy
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
    tu.extract_TTL_trial_start_times(rec_dir, gap_filename, metadata['DIO_port_num'], save_dir=preprocess_dir)
    
    # @TODO: Find spot to break up TTL based on last trial start time + last trial length + buffer in ms
    #last_TTL_based_on_behav_guess = behav_df['TrialEndTimestamp'][-1:].values[0]
    #last TTL based on TTL gaps and timestamps
    # --------------------------------------------------------------------- #
    # Trial start in absolute time from the recording system
    # --------------------------------------------------------------------- #
    TTL_results = load(preprocess_dir + 'TTL_events.npy') 

    # first 0 is before Bpod session, first 1 is first trial, last 0 is end of last trial
    TTL_timestamps_sec = TTL_results['timestamps'][1:]
    TTL_code = TTL_results['TTL_code'][1:]
    gap_lengths = TTL_results['gap_lengths']['gaps']
    recorded_start_ts = TTL_timestamps_sec[TTL_code == 1]
    
    #Find the longest gap between TTL trial start times
    index_gap = np.argmax(np.ediff1d(recorded_start_ts))
    index_last_TTL_start = len(recorded_start_ts)-1
    TTL_biggest_gap = recorded_start_ts[index_gap+1]-recorded_start_ts[index_gap]
    TTL_indices_beh_1 = [0, (index_gap+1)]
    TTL_indices_beh_2 = [(index_gap+1), index_last_TTL_start]

    last_TTL_based_on_TTL_events = recorded_start_ts[-1:][0]
    # @TODO: break up TTL into "halves" for each experiment type 
    # @TODO: run reconcile_TTL_and_behav_trial_start_times on both halves
    # Note: This function doesn't care about absolute Trodes times
    tu.reconcile_TTL_and_behav_trial_start_times(preprocess_dir, TTL_indices_beh_1, preprocess_dir_auditory, behavior_mat_file[0])
    tu.reconcile_TTL_and_behav_trial_start_times(preprocess_dir, TTL_indices_beh_2, preprocess_dir_dc, behavior_mat_file[1])

#%%
if BEHAVIOR:
    bu.create_behavioral_dataframe(preprocess_dir_dc, metadata)
  
  # %% behavior for Detection Confidence
  #Probably will need a for loop to load two behav files and allign them 
if DATA_OBJECT:
    for beh in range(0, 2): 
        if beh == 0:
            _sd = load(preprocess_dir_auditory + 'TrialEvents.npy')
       
            n_trials = _sd['nTrials'] - 1  # throw out last trial (may be incomplete)
       
       
            trialstart_str = 'recorded_TTL_trial_start_time'
            trial_len = _sd[trialstart_str][1:] - _sd[trialstart_str][:-1]
# 
            behav_dict = dict(
                volume = _sd['Custom']['Volume'][:n_trials],
                frequency = _sd['Custom']['Frequency'][:n_trials],
                TrialStartTimestamp = _sd['TrialStartTimestamp'][:n_trials],
                TrialEndTimestamp = _sd['TrialEndTimestamp'][:n_trials],
                TTLTrialStartTime = _sd[trialstart_str][:n_trials],
                TrialLength = trial_len[:n_trials]
            )
            behav_df = pd.DataFrame.from_dict(behav_dict)
       
            dump(behav_df, preprocess_dir_auditory + "behav_df", compress=3)
            
            downsample_dt = 25  # sample period in ms
 #      
            #----------------------------------------
            # Align the spikes with behavior
            #----------------------------------------
            for probe_i in range(1, metadata['n_probes']+1):
                metadata['probe_num'] = probe_i
                
                # load neural data: [number of neurons x time bins in ms]
                spike_mat = load(preprocess_dir + f"probe{probe_i}/" + spike_mat_str_indiv)['spike_mat']
                behav_df = load(preprocess_dir_auditory + 'behav_df')
                
                
                # align spike times to behavioral data timeframe
                # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
                trialwise_spike_mat_start_aligned, _ = trace_utils.trial_start_align(behav_df, spike_mat, sps=1000)
                
                # subsample (bin) data:
                # [n_neurons x n_trials x (-1 means numpy ca lculates: trial_len / dt) x ds]
                # then sum over the dt bins
                n_neurons = trialwise_spike_mat_start_aligned.shape[0]
                n_trials = trialwise_spike_mat_start_aligned.shape[1]
                trial_binned_mat_start_align = trialwise_spike_mat_start_aligned.reshape(n_neurons, n_trials, -1, downsample_dt)
                trial_binned_mat_start_align = trial_binned_mat_start_align.sum(axis=-1)  # sum over bins
                
                results = {'binned_mat': trial_binned_mat_start_align, 'downsample_dt': downsample_dt}
                dump(results, preprocess_dir + f"probe{metadata['probe_num']}/trial_binned_mat_start_align_auditory.npy", compress=3)
                
 #     
        elif beh == 1:
            #TO DO: I can make preprocess_dir[0] and[1] and just loop through it?
            _sd = load(preprocess_dir_dc + 'TrialEvents.npy')

            n_trials = _sd['nTrials'] - 1  # throw out last trial (may be incomplete)
      
            trialstart_str = 'recorded_TTL_trial_start_time'
            trial_len = _sd[trialstart_str][1:] - _sd[trialstart_str][:-1]
      
            behav_dict = dict(
                stimulus_start_time = _sd['Custom']['TrialData']['StimulusStartTime'][:n_trials],
                reward_start_time = _sd['Custom']['TrialData']['RewardStartTime'][:n_trials],
                signal_volume = _sd['Custom']['TrialData']['SignalVolume'][:n_trials],
                MadeChoice = _sd['Custom']['TrialData']['ResponseLeft'][:n_trials],
                PokeCenterStart = _sd['Custom']['TrialData']['ResponseLeft'][:n_trials],
                TrialStartTimestamp = _sd['TrialStartTimestamp'][:n_trials],
                TrialEndTimestamp = _sd['TrialEndTimestamp'][:n_trials],
                TTLTrialStartTime = _sd[trialstart_str][:n_trials],
                TrialLength = trial_len[:n_trials]
            )
            behav_df = pd.DataFrame.from_dict(behav_dict)
      
            dump(behav_df, preprocess_dir_dc + "behav_df", compress=3)
            
            downsample_dt = 25  # sample period in ms
#
            #----------------------------------------
            # Align the spikes with behavior and to specific events
            #----------------------------------------
            n_neurons = 0
            for probe_i in range(1, metadata['n_probes']+1):
                metadata['probe_num'] = probe_i
                probe_save_dir = preprocess_dir + f"probe{probe_i}/"
                
                # load neural data: [number of neurons x time bins in ms]
                spike_mat = load(preprocess_dir + f"probe{probe_i}/" + spike_mat_str_indiv)['spike_mat']
                behav_df = load(preprocess_dir_dc + 'behav_df')
                behav_df['MadeChoice'] = behav_df['MadeChoice'].notnull() # ResponseLeft in behavior/TrialEvents.npy has values 1 - Left, 0 - Right, and NaN - no choice
                
                # align spike times to behavioral data timeframe
                # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
                
                # -------------------------------------------------------- #
                # Chop neuron activity into trials and align to trial start
                # -------------------------------------------------------- #
                # -- This part replaces align_trialwise_spike_times_to_start function because of the path problems
                cbehav_df = behav_df[behav_df['MadeChoice']].reset_index(drop=True)
                # align spike times to behavioral data timeframe
                # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
                trialwise_start_align_spike_mat_in_ms, _ = trace_utils.trial_start_align(cbehav_df, spike_mat, sps=1000)
                assert trialwise_start_align_spike_mat_in_ms.shape[1] == len(cbehav_df)
                
                dump(trialwise_start_align_spike_mat_in_ms, probe_save_dir + 'trialwise_start_align_spike_mat_in_ms', compress=3)
                # --
                trialwise_binned_mat = trialwise_start_align_spike_mat_in_ms
                
                n_probe_neurons, n_trials, _ = trialwise_binned_mat.shape
                n_neurons += n_probe_neurons
                print('Probe', probe_i, 'has', n_probe_neurons, 'neurons with', n_trials, 'trials.')
                
                # ------------------------------------------------------------------------- #
                # Downsample spiking activity, create alignment traces, and save separately
                # ------------------------------------------------------------------------- #
                trace_utils.align_traces_to_task_events(cbehav_df, trialwise_binned_mat, alignment_param_dict, save_dir=probe_save_dir)
                trace_utils.interpolate_traces(cbehav_df, trialwise_binned_mat, interpolation_param_dict, save_dir=probe_save_dir)
                
                
                # -------------------------------------------------------- #
                # Save datapath, behavioral and metadata to data object
                # -------------------------------------------------------- #
                #print('Creating data object...', end='')
                #metadata['nrn_phy_ids'] = joblib.load(probe_save_dir + f"spike_mat_in_ms.npy")['row_cluster_id']
                #data_objs.TwoAFC(probe_save_dir, cbehav_df, metadata).to_pickle()

print('Process Finished.')
