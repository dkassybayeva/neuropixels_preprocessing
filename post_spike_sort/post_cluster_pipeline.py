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
WRITE_METADATA = False
if not WRITE_METADATA:
    rat = 'Nina2'
    date = '20210625'
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
# output directory of the pipeline
if TOY_DATA:
    rec_dir = rec_dir + 'toybase/'
    preprocess_dir = rec_dir
else:
    if WRITE_METADATA:
        metadata = write_session_metadata_to_csv()
    else:
        metadata = load_session_metadata_from_csv(rat, date)
    
    # all probes share same behav & timestamp data, so just set probe_num=1
    metadata['probe_num'] = 1
    session_paths = get_session_path(metadata)
    preprocess_dir = session_paths['preprocess_dir']
    ou.make_dir_if_nonexistent(preprocess_dir)
    rec_dir = session_paths['rec_dir']
    behavior_mat_file = session_paths['behav_dir'] + metadata['behavior_mat_file']
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if not TOY_DATA:
    for probe_i in range(1, metadata['n_probes']+1):
        metadata['probe_num'] = probe_i
        
        # load/create ephys-specific (probe-specific) paths
        session_paths = get_session_path(metadata)
        spike_dir = preprocess_dir + f'probe{probe_i}/'
        ou.make_dir_if_nonexistent(spike_dir)
        if SAVE_INDIVIDUAL_SPIKETRAINS:
            ou.make_dir_if_nonexistent(spike_dir + 'spike_times/')
        
        # process ephys recordings
        tu.create_spike_mat(session_paths['probe_dir'], spike_dir, session_paths['timestamps_dat'], metadata, fs,
                            save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)
    
    gap_filename = tu.find_recording_gaps(session_paths['timestamps_dat'], fs, max_ISI, preprocess_dir)
    
    if metadata['ott_lab']:
        tu.extract_TTL_trial_start_times(rec_dir, gap_filename, metadata['DIO_port_num'], save_dir=preprocess_dir)
        tu.reconcile_TTL_and_behav_trial_start_times(rec_dir, preprocess_dir, behavior_mat_file)
    else:
        tu.convert_TTL_timestamps_to_nbit_events(rec_dir, gap_filename, save_dir=preprocess_dir)
        tu.add_TTL_trial_start_times_to_behav_data(rec_dir, preprocess_dir, behavior_mat_file)

    bu.calc_event_outcomes(preprocess_dir, metadata)

    bu.create_behavioral_dataframe(preprocess_dir)

for probe_i in range(1, metadata['n_probes']+1):
    metadata['probe_num'] = probe_i
    trialwise_binned_mat, cbehav_df = tu.align_trialwise_spike_times_to_start(metadata, preprocess_dir, trace_subsample_bin_size_ms, TOY_DATA=TOY_DATA)
    
    cbehav_df['session'] = metadata['recording_session_id']
    
    data_objs.create_experiment_data_object(preprocess_dir + f"probe{probe_i}/", metadata, trialwise_binned_mat, cbehav_df)