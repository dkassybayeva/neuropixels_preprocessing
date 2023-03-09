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
from os import listdir, path
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump
from scipy.io.matlab import loadmat

# ----------------------------------------------------------------------- #
rat = 'Nina2'
probe = 'probe1'
KS_version = 'kilosort2.5'
fs = 30e3  # sampling frequency
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
combined_session = '20210616_20210618'

combined_dir = path.join('X:/Neurodata', rat, combined_session, probe, KS_version) + '/'

"""
# Phy's clustering results have to be converted to mat file before
# PhySpikes = load(fullfile(kdrive,rat,session,'spikes_per_cluster.mat'))
"""
# Phy's clustering results
cluster_spikes_dict = load(combined_dir + '.phy/spikes_per_cluster.pkl')

"""
# Phy curing table
PhyLabels = tdfread(fullfile(kdrive,rat,session, 'cluster_info.tsv'))
"""
# Phy curing table (cluster metadata)
# index corresponds to the key in cluster_spikes_dict,
# i.e., cluster_spikes_dict[n].size==cluster_label_df.iloc[n]['n_spikes']
cluster_label_df = pd.read_csv(combined_dir + 'cluster_group.tsv', sep="\t")
# Phy cluster_id labelled as 'good'
good_clusters = cluster_label_df.cluster_id[cluster_label_df['group'] == 'good']

"""
# # load KS timestamps (these are indices in reality!) for each spike index
# KSspiketimes = load(fullfile(kdrive,rat,session,'spike_times.mat'))
# KSspiketimes = KSspiketimes.spikeTimes
"""
# load KS timestamps (these are indices in reality!) for each spike index
spike_times_arr = h5py.File(combined_dir + 'spike_times.mat')['spikeTimes'][()][0]

# ----------------------------------------------------------------------- #
session1 = '20210623_121426'
session1_base_dir = path.join('D:/Neurodata', rat, sesion1 + '.rec')

session2 = '20210625_114657'
session2_base_dir = path.join('D:/Neurodata', rat, sesion2 + '.rec')

# load Trodes timestamps
timestamps_dat_1 = path.join(session1_base_dir, session1 + '.kilosort', session1 + '.timestamps.dat')
trodes_timestamps_1 = get_Trodes_timestamps(timestamps_dat_1)
last_sample_sesh1 = trodes_timestamps_1[-1]  # last_sample_sesh1 / fs = time in sec of the last recorded Trodes sample
# last_spike_sesh1_ms = int(np.ceil(last_spike_sesh1 / fs * 1000))

timestamps_dat_2 = path.join(session2_base_dir, session2 + '.kilosort', session2 + '.timestamps.dat')
trodes_timestamps_2 = get_Trodes_timestamps(timestamps_dat_2)

combined_timestamps = np.concatenate(trodes_timestamps_1, trodes_timestamps_2 + last_sample_sesh1)

#behavior files
behav_file_1 = path.join(session1_base_dir, session1 + '.DIO')
behav_file_2 = path.join(session2_base_dir, session2 + '.DIO')
