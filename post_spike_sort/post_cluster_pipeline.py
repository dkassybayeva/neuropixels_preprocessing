# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""

import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.data_objs as data_objs

TOY_DATA = 0

#----------------------------------------------------------------------#
# The following information needs to be filled out and updated for each
# recording session, up to the PIPELINE heading.
#----------------------------------------------------------------------#
fs = 30000.      # sampling frequency of Trodes
max_ISI = 0.001  # max intersample interval (ISI), above which the period
                 # was considered a "gap" in the recording
trace_subsample_bin_size_ms = 25  # sample period in ms
sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces
SAVE_INDIVIDUAL_SPIKETRAINS = True
recording_session_id = 0

#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
DATAPATH = 'D:/Neurodata'
rat_name = 'Nina2'
date = '20210623_121426' # '20210625_114657'
probe_num = 1

# kilosort directory
kilosort_ver = '2.5'
rec_file_path = f"{DATAPATH}/{rat_name}/{date}.rec/"
kilosort_dir = rec_file_path + f"{date}" + ".kilosort" + kilosort_ver + "_probe{}/"
session_path = kilosort_dir.format(probe_num)

# location of Trodes timestamps (in the kilosort folder of first probe)
timestamp_file = rec_file_path + f"{date}" + ".kilosort/" + date + '.timestamps.dat'

# name of the BPod behavioral data file
behavior_mat_file = rec_file_path + "Nina2_Dual2AFC_Jun25_2021_Session1.mat"

# output directory of the pipeline
if TOY_DATA:
    session_path = session_path + 'toybase/'
    output_dir = session_path
else:
    output_dir = session_path + 'preprocessing_output/'
    ou.make_dir_if_nonexistent(output_dir)
    if SAVE_INDIVIDUAL_SPIKETRAINS:
        ou.make_dir_if_nonexistent(output_dir + 'spike_times/')
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#               Create metadata to save with data object               #
#----------------------------------------------------------------------#
metadata = {'time_investment': False,
            'reward_bias': True,
            'prior': False,  # Could possibly be Amy's code for a task type that was previously used
            'experimenter': 'Amy',
            'region': 'lOFC',
            'recording_type': 'neuropixels',
            'experiment_id': 'learning_uncertainty',
            'linking_group': 'Nina2',
            'rat_name': rat_name,
            'date': date,
            'sps': sps,
            'probe_num': probe_num,
            }

#-----------------------------------#
# List of stimuli for each experiment:
#     'freq' =
#     'freq_nat' =
#     'nat' =
#     'nat_nat' =
#-----------------------------------#
metadata['stimulus'] = 'freq'

#-----------------------------------#
# Session number code:
#     -1 = good performance, no noise (or rare)
#      0 = first day with noise
#     -2 = poor performance, no noise (<70%)
#-----------------------------------#
metadata['behavior_phase'] = -1
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if not TOY_DATA:
    tu.create_spike_mat(session_path, output_dir, timestamp_file, date, probe_num, fs,
                        save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)

    gap_filename = tu.find_recording_gaps(timestamp_file, fs, max_ISI, output_dir)

    tu.extract_TTL_events(session_path, gap_filename, save_dir=output_dir)

    tu.add_TTL_trial_start_times_to_behav_data(session_path, output_dir, behavior_mat_file)

    tu.calc_event_outcomes(output_dir)

    tu.create_behavioral_dataframe(output_dir)

trialwise_binned_mat, cbehav_df = tu.align_trialwise_spike_times_to_start(metadata, output_dir, trace_subsample_bin_size_ms, TOY_DATA=TOY_DATA)

cbehav_df['session'] = recording_session_id

data_objs.create_experiment_data_object(session_path, metadata, trialwise_binned_mat, cbehav_df)