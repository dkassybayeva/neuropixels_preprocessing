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
from os import listdir, path
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump
from scipy.io.matlab import loadmat

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 \
    import get_Trodes_timestamps
# ----------------------------------------------------------------------- #


# ------------------------ #
#      Script Params
# ------------------------ #
SEPARATE_SESSION_SPIKETIMES = 0
FIND_COMMON_UNITS = 1
# ------------------------ #


# ----------------------------------------------------------------------- #
#                           Session Info
# ----------------------------------------------------------------------- #
rat = 'Nina2'
probe = 'probe1'
KS_version = 'kilosort2.5'
session1 = '20210623_121426'
session2 = '20210625_114657'
combined_session = '20210623_20210625'
neurodata_dir = 'D:/Neurodata'
combined_dir = f'X:NeuroData/{rat}/{combined_session}/{probe}/{KS_version}/'
fs = 30e3  # sampling frequency in Hz
# ----------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------------------------- #
if SEPARATE_SESSION_SPIKETIMES:
    # ----------------------------------------------------------------------- #
    # Phy's clustering results
    cluster_spikes_dict = load(combined_dir + '.phy/spikes_per_cluster.pkl')

    # Phy curation table - find those curated as good
    # cluster_id corresponds to the key in cluster_spikes_dict
    cluster_label_df = pd.read_csv(combined_dir + 'cluster_group.tsv', sep="\t")
    good_clusters = cluster_label_df.cluster_id[cluster_label_df['group'] == 'good'].to_numpy()

    # load KS timestamps (these are indices in reality!) for each spike index
    spike_times_arr = h5py.File(combined_dir + 'spike_times.mat')['spikeTimes'][()][0]
    # ----------------------------------------------------------------------- #


    # ----------------------------------------------------------------------- #
    def load_session_timestamps(session):
        session_base_dir = path.join(neurodata_dir, rat, session + '.rec')
        timestamps_dat = path.join(session_base_dir, session + '.kilosort', session + '.timestamps.dat')
        trodes_timestamps = get_Trodes_timestamps(timestamps_dat)
        last_sample_sesh = trodes_timestamps[-1]  # last_sample_sesh1 / fs = time in sec of the last recorded Trodes sample
        last_spike_sesh_ms = int(np.ceil(last_sample_sesh / fs * 1000))

        return trodes_timestamps, last_sample_sesh, last_spike_sesh_ms

    trodes_timestamps_1, last_sample_sesh1, last_spike_sesh1_ms = load_session_timestamps(session1)
    trodes_timestamps_2, last_sample_sesh2, last_spike_sesh2_ms = load_session_timestamps(session2)
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
    dump(results, combined_dir + f'spike_mat_in_ms_{rat}_{session1}_{probe}_from_combined_data.npy', compress=3)

    results = {'spike_mat': spike_mat_sesh2, 'row_cluster_id': good_clusters}
    dump(results, combined_dir + f'spike_mat_in_ms_{rat}_{session2}_{probe}_from_combined_data.npy', compress=3)
    print('\nSaved to ' + combined_dir)
# ---------------------------------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------------------------- #
if FIND_COMMON_UNITS:
    DATA_DIR = f'/home/mud/Workspace/ott_neuropix_data/'
    SESSION_DIR = DATA_DIR + f'{rat}/' + '{}/{}' + f'.{KS_version}_{probe}'
    PREPROCESS_DIR = SESSION_DIR + '/preprocessing_output/'
    spike_mat_str_indiv = f'spike_mat_in_ms.npy'
    spike_mat_str_comb = f'spike_mat_in_ms_{rat}_' + '{}' + f'_{probe}_from_combined_data.npy'

    def load_spike_mat_and_row_labels(session, filename):
        session_data = load(PREPROCESS_DIR.format(session, session) + filename)
        spike_mat = session_data['spike_mat']
        unit_list = session_data['row_cluster_id']
        assert spike_mat.shape[0] == len(unit_list)

        return spike_mat, unit_list, len(unit_list)

    spike_mat_sesh1_indiv, units_sesh1_indiv, n_indiv = load_spike_mat_and_row_labels(session1, spike_mat_str_indiv)
    spike_mat_sesh1_common, units_sesh1_common, n_common = load_spike_mat_and_row_labels(session1, spike_mat_str_comb.format(session1))

    """
    Iterate through each row of the common data (which will, in practice, contain fewer 'good' units)
    and look for a spike train that is 'similar' to the common unit spike train.  It is very unlikely, if not 
    impossible, that Kilosort will come up with the same waveform template for the spikes of a unit in one session
    as it does for a single session, meaning that different spike times may be matched, depending on the template.
    
    Therefore, some tolerance needs to be set here:
        - what is the temporal precision?
        - what percentage of matched spikes needs to be found in the individual session?
        
    It is likely that, if each session has a different template being applied to the same spikes of a given unit,
    the spike times registered by one template would have a constant offset from the spike times registered by the other.
    A good measure might be correlation for this, which would show a peak at or around tau=0.  If the correlation
    at any lag is then greater than some threshold, the units are considered to be the same.
    
    First, filter out units which have very different firing rates (sum of spikes) to reduce the computational load.
    Also, make sure to normalize the spike trains, otherwise the correlation peak will be too dependent on spike number.
    """

    n_common_spikes = spike_mat_sesh1_common.sum(axis=1)
    assert n_common_spikes.size == n_common

    n_indiv_spikes = spike_mat_sesh1_indiv.sum(axis=1)
    assert n_indiv_spikes.size == n_indiv

    n_spikes_tol = 1e3  # allow the number of spikes to be different by n_spikes_tol

    raise NotImplementedError

    # @TODO: convolve all trains with some Gaussian before the loop
    spike_mat_sesh1_common_convolved = None
    spike_mat_sesh1_indiv_convolved = None
    assert spike_mat_sesh1_indiv_convolved and spike_mat_sesh1_indiv_convolved
    temporal_precision = 10  # ms; window around tau=0 within which the correlation peak should exceed some threshold
    half_temp = int(0.5 * temporal_precision)
    fraction_threshold = 0.9  # the peak of the correlation should exceed 0.9 of the number of spikes in reference train

    candidate_units_sesh1 = np.empty(n_common, dtype='numpy.nan')

    # st = spiketrain
    for st_i, common_spiketrain in enumerate(spike_mat_sesh1_common_convolved):
        n_common_spikes[st_i]
        correlation_arr = np.zeros(n_indiv)
        for st_j, indiv_spiketrain in enumerate(spike_mat_sesh1_indiv_convolved):
            if np.abs(n_indiv_spikes[st_j] - n_common_spikes[st_i]) > n_spikes_tol:
                continue
            _corr_st = np.correlate(common_spiketrain, indiv_spiketrain, mode='same')
            # @TODO: test normalization
            _corr_st /= (n_indiv_spikes[st_j] * n_common_spikes[st_i])

            # take maximum around tau=0 lag @TODO: test that mid_lag is actually around tau=0
            win_of_interest = _corr_st[mid_lag - half_temp:mid_lag + half_temp]
            correlation_arr[st_j] = win_of_interest.max()

        # the most similar unit is the
        if correlation_arr.max() > fraction_threshold:
            candidate_units_sesh1[st_i] = np.argmax(correlation_arr)


    # @TODO: now do the same thing for the second session

    # @TODO: then see if there are any units that have non-NaN values in both candidate_units_sesh1 and candidate_units_sesh2
    # @TODO: if so, then save these mappings, which connect the unit # in session 1 with the unit # in session 2

# ---------------------------------------------------------------------------------------------------------- #
