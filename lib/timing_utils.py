"""
Make spike time vectors for single unit recordings 
with Neuropixels & Trodes

convert KS2.5 clustering results, cured in Phy, to spike times 
using Trode's timestamps

Created on Thu Dec  8 03:54:44 2022

@author: Greg Knoll
"""
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile


def create_spike_mat(session_path, kilosort_dir, date, probe_num, fs):
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
    # load Trodes timestamps - in the general kilosort folder (of first probe)
    timestamp_file = kilosort_dir.format(1) + date + '.timestamps.dat'
    # >1GB variable for a 3h recording
    trodes_timestamps_tuples = readTrodesExtractedDataFile(timestamp_file)['data']
    trodes_timestamps = np.array([t[0] for t in trodes_timestamps_tuples])
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
         
        # save spike times
        spike_time_file = f'spike_times_in_sec_shank={probe_num}_clust={clust_i}.npy'
        dump(spike_train, cellbase_dir + spike_time_file, compress=3)    
        
        # register spikes in the spike matrix
        spiktime_ms_inds = np.round(spike_train * 1000).astype('int')
        spike_mat[i, spiktime_ms_inds] = 1
    
        
    dump(spike_mat, cellbase_dir + 'spike_mat.npy', compress=3)
    print('\nSaved to ' + cellbase_dir)


def calc_intersample_periods(timestamps, fs=30000., threshold=0.001, save_dir=''):
    """
    Returns lengths and start times of intersample periods.

    Parameters
    ----------
    timestamps : [numpy array] trodes timestamps.
    fs : [float] sampling frequency. The default is 30000.
    threshold : [float] size of gap in seconds to keep. 
                The default is 0.001 (1ms).

    Returns
    -------
    None.  Saves the results to gaps

    """
    # length of gaps
    gaps = np.diff(timestamps) / fs
    
    # gaps_ts are timestamps where gaps *start*
    gaps_ts = timestamps[gaps > threshold] / fs
    
    gaps = gaps[gaps > threshold]
    gaps_ts = gaps_ts[gaps > threshold]
    
    # also save some info for later in cellbase folder
    results = {'gaps': gaps, 'gaps_ts': gaps_ts}
    gap_filename = f"trodes_intersample_periods_longer_than_{threshold}s.npy"
    dump(results, save_dir + gap_filename, compress=3)
    
    return gap_filename


def process_behavioral_events(trodes_timestamps, fs, threshold, session_path):
    """
    Convert Trodes Analog Input to TTL Events

    Parameters
    ----------
    trodes_timestamps : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.
    session_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    save_dir = session_path + 'cellbase/'
    gap_filename = calc_intersample_periods(trodes_timestamps, fs, threshold, save_dir)
    events_TTL, events_TS = extractTTLs(session_path, gap_filename)
    results = {'TTL': events_TTL, 'TS': events_TS}
    dump(results, save_dir + 'events.npy', compress=3) 
