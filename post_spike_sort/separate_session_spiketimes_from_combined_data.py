"""
Script run independently from the python preprocessing pipeline, which
was intended for individual sessions.  This script is used to assign
the spike times from a curated, combined session into their corresponding
original sessions:

1. Kilosort was run on the combined raw data, and then the sorted units
were checked to ensure their first principal component was consistent at
the session intersection; these are labelled as 'good' in Phy.

2. The spike times for each good unit are then separated into two arrays
(before and after the session intersection) and added to the corresponding
sessions' matrices.  This second step is only a slight modification of
timing_utils:create_spike_mat() used for the individual sessions.

Author: Greg Knoll
Date: March 2023
"""
# ----------------------------------------------------------------------- #
from os import path
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 \
    import get_Trodes_timestamps
from neuropixels_preprocessing.session_params import *
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
# Phy's clustering results
cluster_spikes_dict = load(STITCH_DIR + '.phy/spikes_per_cluster.pkl')

# Phy curation table - find those curated as good
# cluster_id corresponds to the key in cluster_spikes_dict
cluster_label_df = pd.read_csv(STITCH_DIR + 'cluster_group.tsv', sep="\t")
good_clusters = cluster_label_df.cluster_id[cluster_label_df['group'] == 'good'].to_numpy()

# load KS timestamps (these are indices in reality!) for each spike index
spike_times_arr = h5py.File(STITCH_DIR + 'spike_times.mat')['spikeTimes'][()][0]
# ----------------------------------------------------------------------- #


print('------------------------------------------------------------')
print('      Separating Session Spiketimes from Combined Data      ')
print('------------------------------------------------------------')


# ----------------------------------------------------------------------- #
def load_session_timestamps(session, session_dir):
    timestamps_dat = path.join(session_dir + '.kilosort', session + '.timestamps.dat')
    trodes_timestamps = get_Trodes_timestamps(timestamps_dat)
    last_sample_sesh = trodes_timestamps[-1]  # last_sample_sesh1/fs = time [sec] of the last recorded Trodes sample
    last_spike_sesh_ms = int(np.ceil(last_sample_sesh / fs * 1000))

    return trodes_timestamps, last_sample_sesh, last_spike_sesh_ms


trodes_timestamps_1, last_sample_sesh1, last_spike_sesh1_ms = load_session_timestamps(session1, REC_PATH)
trodes_timestamps_2, last_sample_sesh2, last_spike_sesh2_ms = load_session_timestamps(session2, REC_PATH2)
combined_timestamps = np.concatenate((trodes_timestamps_1, trodes_timestamps_2 + last_sample_sesh1))
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
# data structures for spiking data
spike_mat_sesh1 = np.zeros((len(good_clusters), last_spike_sesh1_ms), dtype='uint8')
spike_mat_sesh2 = np.zeros((len(good_clusters), last_spike_sesh2_ms), dtype='uint8')
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
print('Creating spike mat...')
for i, clust_i in enumerate(good_clusters):
    print(f'{i + 1} / {len(good_clusters)}\r', flush=True, end='')

    # cluster's spike indices in the full KS spike_times_arr
    clust_spike_idx = cluster_spikes_dict[clust_i]

    # use the indices to find the corresponding Trode samples
    clust_spike_sample_idx = spike_times_arr[clust_spike_idx]

    # Convert relative spike sample indices to global sample indices.
    # Because Trodes timestamps are supposed to be just consecutive integers,
    # the same could be achieved potentially by simply adding trodes_timestamps[0]
    # to clust_spike_sample_idx, but there are sometimes missing samples, such that
    # the sample numbers aren't always consecutive.
    # To be safe, index the Trodes samples in this way.
    global_clust_spike_sample_idx = combined_timestamps[clust_spike_sample_idx]

    spike_samples_sesh1 = global_clust_spike_sample_idx[global_clust_spike_sample_idx <= last_sample_sesh1]
    spike_samples_sesh2 = global_clust_spike_sample_idx[global_clust_spike_sample_idx > last_sample_sesh1]

    # Trodes saves timestamp as index in sampling frequency
    spike_train_sesh1 = spike_samples_sesh1 / fs  # actual spike times in seconds
    spike_train_sesh2 = (spike_samples_sesh2 - last_sample_sesh1) / fs  # actual spike times in seconds

    # register spikes in the spike matrix
    spiktime_ms_inds_sesh1 = np.round(spike_train_sesh1 * 1000).astype('int')
    spike_mat_sesh1[i, spiktime_ms_inds_sesh1] = 1

    spiktime_ms_inds_sesh2 = np.round(spike_train_sesh2 * 1000).astype('int')
    spike_mat_sesh2[i, spiktime_ms_inds_sesh2] = 1

results = {'spike_mat': spike_mat_sesh1, 'row_cluster_id': good_clusters}
dump(results, STITCH_DIR + f'spike_mat_in_ms_{rat}_{session1}_{probe}_from_combined_data.npy', compress=3)

results = {'spike_mat': spike_mat_sesh2, 'row_cluster_id': good_clusters}
dump(results, STITCH_DIR + f'spike_mat_in_ms_{rat}_{session2}_{probe}_from_combined_data.npy', compress=3)
print('\nSaved to ' + STITCH_DIR)
# ----------------------------------------------------------------------- #
