# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""
import os
import numpy as np
from tqdm import trange
import joblib

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou
import neuropixels_preprocessing.lib.behavior_utils as bu
import neuropixels_preprocessing.lib.data_objs as data_objs
import neuropixels_preprocessing.lib.trace_utils as trace_utils
from neuropixels_preprocessing.session_params import *


#----------------------------------------------------------------------#
# The information in the metadata block of session_params needs to be
# filled out and updated for each recording session.
#----------------------------------------------------------------------#
DATA_ROOT = 'server'  # ['local', 'server', 'X:', etc.]
SPIKES_AND_TTL = False
BEHAVIOR = False
LFPs = False
DATA_OBJECT = False

SAVE_INDIVIDUAL_SPIKETRAINS = True
WRITE_METADATA = False
    
if WRITE_METADATA:
    metadata = write_session_metadata_to_csv(DATA_ROOT)
else:
    rat = 'Nina2'
    date = '20210623'
    metadata = load_session_metadata_from_csv(DATA_ROOT, rat, date)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
metadata['probe_num'] = '' # don't need probe dir for now
session_paths = get_session_path(metadata, DATA_ROOT, is_ephys_session=True)
preprocess_dir = session_paths['preprocess_dir']
ou.make_dir_if_nonexistent(preprocess_dir)
rec_dir = session_paths['rec_dir']
behavior_mat_file = session_paths['behav_dir'] + metadata['behavior_mat_file']
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#                           PIPELINE                                   #
#----------------------------------------------------------------------#
if SPIKES_AND_TTL:
    for probe_i in range(1, metadata['n_probes']+1):
        metadata['probe_num'] = probe_i

        # load/create ephys-specific (probe-specific) paths
        session_paths = get_session_path(metadata, DATA_ROOT, is_ephys_session=True)
        spike_dir = preprocess_dir + f'probe{probe_i}/'
        ou.make_dir_if_nonexistent(spike_dir)
        if SAVE_INDIVIDUAL_SPIKETRAINS:
            ou.make_dir_if_nonexistent(spike_dir + 'spike_times/')

        # process ephys recordings
        tu.create_spike_mat(session_paths['probe_dir'], spike_dir, session_paths['timestamps_dat'], metadata, fs,
                            save_individual_spiketrains=SAVE_INDIVIDUAL_SPIKETRAINS)

    tu.find_recording_gaps(session_paths['timestamps_dat'], fs, max_ISI, preprocess_dir + gap_filename)

    if metadata['ott_lab']:
        tu.extract_TTL_trial_start_times(session_paths['probe_dir'], gap_filename, metadata['DIO_port_num'], save_dir=preprocess_dir)
        tu.reconcile_TTL_and_behav_trial_start_times(rec_dir, preprocess_dir, behavior_mat_file)
    else:
        tu.convert_TTL_timestamps_to_nbit_events(rec_dir, gap_filename, save_dir=preprocess_dir)
        tu.add_TTL_trial_start_times_to_behav_data(rec_dir, preprocess_dir, behavior_mat_file)


if BEHAVIOR:
    bu.create_behavioral_dataframe(preprocess_dir, metadata)


if LFPs:
    # -------------------------------------------------------- #
    #               Find LFPs for a given session
    #             (Common directory for both probes)
    # -------------------------------------------------------- #
    LFP_dir = session_paths['rec_dir']+metadata['trodes_datetime']+'.LFP/'
    assert os.path.exists(LFP_dir)
    LFP_dat_l = np.sort(os.listdir(LFP_dir))
    LFP_file_str = LFP_dat_l[0].split('nt')[0] + 'nt' + '{}' + 'ch1.dat'

    for probe_i in range(1, metadata['n_probes']+1):
        # -------------------------------------------------------- #
        #              Probe-specific output directory
        # -------------------------------------------------------- #
        metadata['probe_num'] = probe_i
        session_paths = get_session_path(metadata, DATA_ROOT, is_ephys_session=True)
        lfp_output_dir = preprocess_dir + f'probe{probe_i}/' + 'LFPs/'
        ou.make_dir_if_nonexistent(lfp_output_dir)

        # -------------------------------------------------------- #
        #      Export LFP data from channels of good units
        # -------------------------------------------------------- #
        cluster_label_df = pd.read_csv(session_paths['probe_dir'] + 'cluster_info.tsv', sep="\t")
        good_cluster_channels = np.unique(cluster_label_df.ch[cluster_label_df['group'] == 'good'].to_numpy())

        for chan_i in trange(len(good_cluster_channels)):
            chan = good_cluster_channels[chan_i]
            ch_lfp_data = readTrodesExtractedDataFile(LFP_dir + LFP_file_str.format(str(1000 * probe_i + chan)))['data']
            ch_lfp_data_arr = np.array([x[0] for x in ch_lfp_data])
            joblib.dump(ch_lfp_data_arr, lfp_output_dir + f'ch{chan}',compress=5)


if DATA_OBJECT:
    n_neurons = 0
    for probe_i in range(1, metadata['n_probes']+1):
        metadata['probe_num'] = probe_i

        # -------------------------------------------------------- #
        # Chop neuron activity into trials and align to trial start
        # -------------------------------------------------------- #
        probe_save_dir = preprocess_dir + f"probe{probe_i}/"
        trialwise_binned_mat, cbehav_df = tu.align_trialwise_spike_times_to_start(preprocess_dir, probe_save_dir)

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
        print('Creating data object...', end='')
        metadata['nrn_phy_ids'] = joblib.load(probe_save_dir + f"spike_mat_in_ms.npy")['row_cluster_id']
        data_objs.TwoAFC(probe_save_dir, cbehav_df, metadata).to_pickle()


if WRITE_METADATA:
    insert_value_into_metadata_csv(DATA_ROOT, rat, date, 'n_good_units', n_neurons)
    insert_value_into_metadata_csv(DATA_ROOT, rat, date, 'n_trials', n_trials)
