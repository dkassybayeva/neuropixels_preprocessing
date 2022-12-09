"""
Make spike time vectors for single unit recordings 
with Neuropixels & Trodes

convert KS2.5 clustering results, cured in Phy, to spike times 
using Trode's timestamps

Created on Thu Dec  8 03:54:44 2022

@author: Greg Knoll
"""
from os import listdir
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 \
    import get_Trodes_timestamps, readTrodesExtractedDataFile


def create_spike_mat(session_path, timestamp_file, date, probe_num, fs,
                     save_individual_spiketrains):
    #----------------------------------------------------------------------#
    #                   Load Kilosort/Phy Spike Data
    #----------------------------------------------------------------------#
    # Phy's clustering results
    cluster_spikes_dict = load(session_path + '.phy/spikes_per_cluster.pkl')
    
    # load KS timestamps (these are indices in reality!) for each spike index
    spike_times_arr = h5py.File(session_path + 'spike_times.mat')['spikeTimes'][()][0]
    #----------------------------------------------------------------------#
    
    
    #----------------------------------------------------------------------#
    #              Load Trodes Times for Relative Timekeeping
    #----------------------------------------------------------------------#
    # >1GB variable for a 3h recording
    trodes_timestamps = get_Trodes_timestamps(timestamp_file)
    #----------------------------------------------------------------------#
    
    
    #----------------------------------------------------------------------#
    #                   Get indices of good units
    #----------------------------------------------------------------------#
    # Phy curing table (cluster metadata)
    # index corresponds to the key in cluster_spikes_dict, 
    # i.e., cluster_spikes_dict[n].size==cluster_label_df.iloc[n]['n_spikes']
    cluster_label_df = pd.read_csv(session_path + 'cluster_info.tsv', sep="\t")
    
    # Phy cluster_id labelled as 'good'
    good_clusters = cluster_label_df.cluster_id[cluster_label_df['group'] =='good']
    #----------------------------------------------------------------------#
    
    
    #----------------------------------------------------------------------#
    #             Create Spike Time Vectors and Save in Matrix
    #----------------------------------------------------------------------#
    cellbase_dir = session_path + 'cellbase/'
    
    # Create a matrix with a row for each good cluster and all rows same length
    last_spike_in_sec = trodes_timestamps[-1] / fs
    last_spike_ms = int(np.ceil(last_spike_in_sec * 1000))
    spike_mat = np.zeros((len(good_clusters), last_spike_ms))
    
    print('Creating spike mat...')
    for i, clust_i in enumerate(good_clusters):
        print(f'{i+1} / {len(good_clusters)}\r', flush=True, end='')
        
        # spike indices of cluster
        clust_spike_idx = cluster_spikes_dict[clust_i]  
        
        # spike index to KiloSort time index
        clust_spike_times = spike_times_arr[clust_spike_idx]  
    
        # KiloSort time index to sample index in Trodes
        global_spike_times = trodes_timestamps[clust_spike_times] 
        
        # Trodes saves timestamp as index in sampling frequency
        spike_train = global_spike_times / fs  # actual spike times in seconds
         
        if save_individual_spiketrains:
            # save spike times
            spike_time_file = f'spike_times_in_sec_shank={probe_num}_clust={clust_i}.npy'
            dump(spike_train, cellbase_dir + spike_time_file, compress=3)    
        
        # register spikes in the spike matrix
        spiktime_ms_inds = np.round(spike_train * 1000).astype('int')
        spike_mat[i, spiktime_ms_inds] = 1
    
        
    dump(spike_mat, cellbase_dir + 'spike_mat.npy', compress=3)
    print('\nSaved to ' + cellbase_dir)


def find_recording_gaps(timestamp_file, fs, max_ISI, save_dir):
    """
    Detects abnormalities in the lengths of the periods between samples which
    may result from the recording device temporarily going offline.

    Parameters
    ----------
    timestamps : [numpy array] trodes timestamps.
    fs : [float] sampling frequency. The default is 30000.
    max_ISI : [float] largest period between samples (ISI=intersample interval)
              allowed before the period is considered a "gap" in the recording
              The default is 0.001 (1ms).

    Returns
    -------
    None.  Saves the results to gaps

    """
    trodes_timestamps = get_Trodes_timestamps(timestamp_file)
    
    # length of gaps
    gaps = np.diff(trodes_timestamps) / fs
    
    # gaps_ts are timestamps where gaps *start*
    gaps_ts = trodes_timestamps[:-1][gaps > max_ISI] / fs
    
    gaps = gaps[gaps > max_ISI]
    gaps_ts = gaps_ts[gaps > max_ISI]
    
    # also save some info for later in cellbase folder
    results = {'gaps': gaps, 'gaps_ts': gaps_ts}
    gap_filename = f"trodes_intersample_periods_longer_than_{max_ISI}s.npy"
    dump(results, save_dir + gap_filename, compress=3)
    
    return gap_filename


def extract_TTL_events(session_path, gap_filename, save_dir):
    """
    Converts 6 analog channels to TTL-like event times in seconds
    TTL = transistor-transistor logic

    Requires export of .DIO in Trodes.
    
    Original Matlab version: LC/QC/TO 2018-21
    Ported to python: Greg Knoll 2022
    
    Parameters
    ----------
    fname is a .rec file (full path)
    
    Returns
    -------
    Events_TTL, Events_TS
    """
         
    dio_path = '.'.join(session_path.split('.')[:-1]) + '.DIO/'
    
    # each analog MCU input pin will have its own .dat file
    dio_file_list = listdir(dio_path)
    n_channels = len(dio_file_list)
    
    
    # ------------------------------------------------------------ #
    #           Load and organize TTL timestamps and states
    # ------------------------------------------------------------ #
    TTL_timestamps = np.array([])
    timestamp_list = []
    state_list = []
    for din_filename in dio_file_list:
        if not('Din' in din_filename and '.dat' in din_filename):
            continue
        
        # Load the channel dictionary: data + metadata
        channel_dict = readTrodesExtractedDataFile(dio_path + din_filename)
        if not channel_dict:
            print('Error while trying to read ' + din_filename)
            continue
        
        # Each data point is (timestamp, state) -> break into separate arrays
        channel_data = channel_dict['data']
        channel_states = np.array([tup[1] for tup in channel_data])
        channel_timestamps = np.array([tup[0] for tup in channel_data])
        assert channel_states.shape == channel_timestamps.shape
        
        # Convert timestamps to seconds and save both structures in their
        # respective containers
        ch_timestamps_sec = channel_timestamps / int(channel_dict['clockrate'])
        TTL_timestamps = np.append(TTL_timestamps, ch_timestamps_sec)
        timestamp_list.append(ch_timestamps_sec)
        state_list.append(channel_states)
        
    assert sum(map(len, state_list)) == TTL_timestamps.size
    assert sum(map(len, timestamp_list)) == TTL_timestamps.size
    # ------------------------------------------------------------ #
    
    
    # ------------------------------------------------------------ #
    #  Create n-channel-bit code with length as long as unique timestamps
    # ------------------------------------------------------------ #
    TTL_timestamps = np.unique(TTL_timestamps)
    states_mat = np.zeros((n_channels, TTL_timestamps.size))
    
    # Register the state of each channel at each global timestamp
    # and interpolate between timestamps
    for ch_i in range(n_channels):
        # TODO: this could probably be done without iteration, e.g.:
        # idx = np.nonzero(np.in1d(TTL_timestamps, timestamp_list[ch_i]).sum())[0]
        for ts_i in range(TTL_timestamps.size):
             
            idx = np.where((timestamp_list[ch_i] == TTL_timestamps[ts_i]))[0]
            if idx.size:  # state switch
                states_mat[ch_i, ts_i] = state_list[ch_i][idx]
            else:  # interpolate
                states_mat[ch_i, ts_i] = states_mat[ch_i, ts_i-1]

        coding_bit = 2**ch_i  # Each channel represents particular bit in code
        states_mat[ch_i] = states_mat[ch_i] * coding_bit  
        
    TTL_code = np.sum(states_mat, axis=0)  # Convert to n-channel-bit code
    assert TTL_code.size == TTL_timestamps.size
    # ------------------------------------------------------------ #
    

    # ------------------------------------------------------------ #
    #           Now consider gaps in the recordings
    # ------------------------------------------------------------ #
    gaps = load(save_dir + gap_filename)
    gap_timestamps = gaps['gaps_ts']
    
    # append their timestamps to the timestamps array
    TTL_timestamps = np.append(TTL_timestamps, gap_timestamps)
    
    # add -1 as placeholder code for the gaps
    TTL_code = np.append(TTL_code, -1 * np.ones(gap_timestamps.size))
    assert TTL_code.size == TTL_timestamps.size
    
    # resort the timestamps
    sort_idx = np.argsort(TTL_timestamps)
    TTL_timestamps = TTL_timestamps[sort_idx]
    TTL_code = TTL_code[sort_idx]
    # ------------------------------------------------------------ #
    
    # ------------------------------------------------------------ #
    #                      Save results
    # ------------------------------------------------------------ #
    results = {'TTL_code': TTL_code, 'timestamps': TTL_timestamps}
    dump(results, save_dir + 'TTL_events.npy', compress=3) 