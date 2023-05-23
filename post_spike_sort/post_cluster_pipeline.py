# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""

import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.data_objs as data_objs
from neuropixels_preprocessing.session_params import *

TOY_DATA = 0  # for testing

#----------------------------------------------------------------------#
# The information in this block needs to be filled out and updated
# for each recording session.
#----------------------------------------------------------------------#
SAVE_INDIVIDUAL_SPIKETRAINS = True

max_ISI = 0.001  # max intersample interval (ISI), above which the period
                 # was considered a "gap" in the recording
trace_subsample_bin_size_ms = 25  # sample period in ms
sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces
metadata['sps'] = sps
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
# output directory of the pipeline
if TOY_DATA:
    SESSION_DIR = SESSION_DIR + 'toybase/'
    output_dir = SESSION_DIR
else:
    output_dir = SESSION_DIR + 'preprocessing_output/'
    ou.make_dir_if_nonexistent(output_dir)
    if SAVE_INDIVIDUAL_SPIKETRAINS:
        ou.make_dir_if_nonexistent(output_dir + 'spike_times/')
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if not TOY_DATA:
    tu.create_spike_mat(SESSION_DIR, output_dir, timestamps_dat, session1, probe, fs,
                        save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)

    gap_filename = tu.find_recording_gaps(timestamps_dat, fs, max_ISI, output_dir)

    if OTT_LAB_DATA:
        tu.extract_TTL_trial_start_times(SESSION_DIR, gap_filename, metadata['DIO_port_num'], save_dir=output_dir)
    else:
        tu.convert_TTL_timestamps_to_nbit_events(SESSION_DIR, gap_filename, save_dir=output_dir)

    tu.add_TTL_trial_start_times_to_behav_data(SESSION_DIR, output_dir, REC_PATH+behavior_mat_file)

    tu.calc_event_outcomes(output_dir)

    tu.create_behavioral_dataframe(output_dir)

trialwise_binned_mat, cbehav_df = tu.align_trialwise_spike_times_to_start(metadata, output_dir, trace_subsample_bin_size_ms, TOY_DATA=TOY_DATA)

cbehav_df['session'] = recording_session_id

data_objs.create_experiment_data_object(SESSION_DIR, metadata, trialwise_binned_mat, cbehav_df)