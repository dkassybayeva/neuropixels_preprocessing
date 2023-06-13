# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""

import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.behavior_utils as bu
import neuropixels_preprocessing.lib.data_objs as data_objs
from neuropixels_preprocessing.session_params import *

TOY_DATA = 0  # for testing

#----------------------------------------------------------------------#
# The information in the metadata block of session_params needs to be
# filled out and updated for each recording session.
#----------------------------------------------------------------------#
SAVE_INDIVIDUAL_SPIKETRAINS = True
#----------------------------------------------------------------------#

#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
# output directory of the pipeline
if TOY_DATA:
    session_dir = session_dir + 'toybase/'
    preprocess_dir = session_dir
else:
    metadata = write_session_metadata_to_csv()
    session_dir, behav_dir, preprocess_dir, timestamps_dat = get_session_path(metadata)
    if SAVE_INDIVIDUAL_SPIKETRAINS:
        ou.make_dir_if_nonexistent(preprocess_dir + 'spike_times/')
    behavior_mat_file = behav_dir + metadata['behavior_mat_file']
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if not TOY_DATA:
    tu.create_spike_mat(session_dir, preprocess_dir, timestamps_dat, metadata, fs,
                        save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)

    gap_filename = tu.find_recording_gaps(timestamps_dat, fs, max_ISI, preprocess_dir)

    if OTT_LAB_DATA:
        tu.extract_TTL_trial_start_times(session_dir, gap_filename, metadata['DIO_port_num'], save_dir=preprocess_dir)
        tu.reconcile_TTL_and_behav_trial_start_times(session_dir, preprocess_dir, behavior_mat_file)
    else:
        tu.convert_TTL_timestamps_to_nbit_events(session_dir, gap_filename, save_dir=preprocess_dir)
        tu.add_TTL_trial_start_times_to_behav_data(session_dir, preprocess_dir, behavior_mat_file)

    bu.calc_event_outcomes(preprocess_dir, metadata)

    bu.create_behavioral_dataframe(preprocess_dir)

trialwise_binned_mat, cbehav_df = tu.align_trialwise_spike_times_to_start(metadata, preprocess_dir, trace_subsample_bin_size_ms, TOY_DATA=TOY_DATA)

cbehav_df['session'] = metadata['recording_session_id']

data_objs.create_experiment_data_object(preprocess_dir, metadata, trialwise_binned_mat, cbehav_df)