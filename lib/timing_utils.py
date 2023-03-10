"""
Library for processing timing events recorded by Trodes from 
Neuropixels probes and Bpod protocols

Created on Thu Dec  8 03:54:44 2022

@author: Greg Knoll
"""
from os import listdir
import numpy as np
import pandas as pd
import h5py
from joblib import load, dump
from scipy.io.matlab import loadmat

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 \
    import get_Trodes_timestamps, readTrodesExtractedDataFile

import neuropixels_preprocessing.lib.trace_utils as trace_utils
import neuropixels_preprocessing.lib.behavior_utils as bu

def create_spike_mat(session_path, timestamp_file, date, probe_num, fs,
                     save_individual_spiketrains):
    """
    Make spiking matrix (and optionally individual spike time vectors, or 
    spike trains) for single unit recordings from Neuropixels & Trodes, 
    after clustering by Kilosort and manual curation by Phy.

    Converts KS2.5 clustering results, cured in Phy, to spike times 
    using Trode's timestamps.

    Parameters
    ----------
    session_path : TYPE
        DESCRIPTION.
    timestamp_file : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.
    probe_num : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    save_individual_spiketrains : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    output_dir = session_path + 'preprocess_pipeline_output/'
    
    # Create a matrix with a row for each good cluster and all rows same length
    last_spike_in_sec = trodes_timestamps[-1] / fs
    last_spike_ms = int(np.ceil(last_spike_in_sec * 1000))
    spike_mat = np.zeros((len(good_clusters), last_spike_ms), dtype='uint8')
    
    print('Creating spike mat...')
    for i, clust_i in enumerate(good_clusters):
        print(f'{i+1} / {len(good_clusters)}\r', flush=True, end='')
        
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
        global_clust_spike_sample_idx = trodes_timestamps[clust_spike_sample_idx]
        
        # Trodes saves timestamp as index in sampling frequency
        spike_train = global_clust_spike_sample_idx / fs  # actual spike times in seconds
         
        if save_individual_spiketrains:
            # save spike times
            spike_time_file = f'spike_times_in_sec_shank={probe_num}_clust={clust_i}.npy'
            dump(spike_train, output_dir + spike_time_file, compress=3)    
        
        # register spikes in the spike matrix
        spiktime_ms_inds = np.round(spike_train * 1000).astype('int')
        spike_mat[i, spiktime_ms_inds] = 1
    
    results = {'spike_mat': spike_mat, 'row_cluster_id': good_clusters}
    dump(results, output_dir + 'spike_mat_in_ms.npy', compress=3)
    print('\nSaved to ' + output_dir)


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
    
    # also save some info for later in output folder
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
    session_path is the .rec directory (full path)
    
    Saves
    -------
    TTL codes as time series with accompanying timestamps in 'TTL_events.npy'
    in the directory save_dir
    """
         
    dio_path = '.'.join(session_path.split('.')[:-2]) + '.DIO/'
    
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


def group_codes_and_timestamps_by_trial(TTL_code, timestamps):
    """
    Group TTLs and their timestamps by trial. 
    This is tricky, since in TRODES we (TO/2020-2021) are using 5 bits to
    sync Bpod events, which are the first 5 DIO at the Trodes MCU. The 6th
    bit is used for laser pulse alignment. 
    However, there can be more than 2^5 Bpod states.
    
    Parameters
    ----------
    TTL_code, Events_TS [numpy arrays]
    
    Returns
    -------
    list with length n_trials, in which each element is a dictionary:
        'TTL_code': list of TTL codes (states) encountered in the trial
        'timestamps': list of timestampts of the codes (states)
    """

    start_code = 1  # trialStart (WaitForInitialPoke)
    pre_start = 0  # no state (between trials)
    post_start = 2  # state that is always followed by start_code state
    
    first_start = np.where(TTL_code==start_code)[0][0]
    n_codes = len(TTL_code)
    
    # Break Nlx events into a cell array by trials
    n_trials = 0
    events_and_timestamps_per_trial = []
    
    # create new data structures for trial's events
    curr_trial_codes = []
    curr_trial_timestamps = []
    for x in range(first_start, n_codes):
        # if a trial start
        if TTL_code[x] == start_code and TTL_code[x-1]==pre_start and TTL_code[x+1]==post_start:
            n_trials = n_trials + 1
            if x != first_start:
                # save previous trial's events to global container
                events_and_timestamps_per_trial.append(
                                    {'TTL_code': curr_trial_codes, 
                                     'timestamps': curr_trial_timestamps})
            
            # create new data structures for trial's events
            curr_trial_codes = []
            curr_trial_timestamps = []
        elif TTL_code[x]==1:  # DUPLICATE STATE CODE!!!
            # This state is also 1, but it is not a start code, 
            # because it is not preceded by 0 and followed by 2.
            # Now that we have access to more bits, correct this state to
            # 2^5+1=33 (Corrected value is SPECIFIC TO STATE MATRIX!)
            TTL_code[x] = 33
            
        # append event to current trial's events and timestamps
        curr_trial_codes.append(TTL_code[x])
        curr_trial_timestamps.append(timestamps[x])
    events_and_timestamps_per_trial.append({'TTL_code': curr_trial_codes, 
                                        'timestamps': curr_trial_timestamps})

    assert len(events_and_timestamps_per_trial) == n_trials
        
    return events_and_timestamps_per_trial


def align_TTL_events(trialwise_TTLs, align_idx=1, save=('', False)):
    """
    Set TTL alignement state to the start_code (1=WaitingForInitialPoke).
    """
    start_times = []
    
    for tw_TTL in trialwise_TTLs:
        start_idx = int(np.where(np.array(tw_TTL['TTL_code']) == align_idx)[0])
        start_time = tw_TTL['timestamps'][start_idx]
        tw_TTL['aligned_timestamps'] = tw_TTL['timestamps'] - start_time
        tw_TTL['start_time'] = start_time
        start_times.append(start_time)
     
    if save[0]:
        dump(trialwise_TTLs, save[1] + 'aligned_TTL_events.npy', compress=3)
    
    return trialwise_TTLs, start_times


def remove_laser_trials(trialwise_TTLs, start_times, laser_TTL=526):
    """
    Remove laser trials (tagging protocol) 
    using specific hard-codee TTL code 526 (NLX system)
    """

    laser_trials = []

    for i, tw_TTL in enumerate(trialwise_TTLs):
        if laser_TTL in tw_TTL['TTL_code']: 
            laser_trials.append(i)
            del trialwise_TTLs[i]
            del start_times[i]
   
    if len(laser_trials):
        print(f'Removed {len(laser_trials)} laser trials: {laser_trials}')
            
    return trialwise_TTLs, start_times


def add_TTL_trial_start_times_to_behav_data(session_dir, behavior_mat_file):
    """
    Synchronize trial events to recording times.
    
    Requires: TTL_events.npy from extract_TTL_events().
    
    MakeTrialEvents2_TorbenNP(SESSIONPATH) loads TRODES events and adjusts
    trial event times (trial-based behavioral data, see
    SOLO2TRIALEVENTS4_AUDITORY_GONOGO) to the recorded time stamps. This
    way the neural recordings and behavioral time stamps are in register.
    Stimulus time TTL pulses are used for synchronization. The synchronized
    trial events structure is saved under the name 'TrialEvents.mat'. This
    file becomes the primary store of behavioral data for a particular
    session it is retrieved by LOADCB via CELLID2FNAMES. This default
    file name is one of the preference settings of CellBase - type 
    getpref('cellbase','session_filename')

    MakeTrialEvents2_TorbenNP(SESSIONPATH,'StimNttl',TTL) specifies the TTL
    channel which serves as the basis for synchronization.
    """

    output_dir = session_dir + 'preprocess_pipeline_output/'
    # --------------------------------------------------------------------- #
    # Trial start time recorded by the recording system (Neuralynx)
    # --------------------------------------------------------------------- #
    # Load converted TRODES event file 
    print('Grouping TTL events by trial and getting recorded trial start times...')
    try:
        TTL_results = load(output_dir + 'TTL_events.npy')
    except:
        print('TTL_events.npy file not found.  Make sure the TTL events \n\
        have been extraced from the TRODES .DIO files.')

    trialwise_TTLs = group_codes_and_timestamps_by_trial(**TTL_results)
    
    aligned_trialwise_TTLs, recorded_start_ts = align_TTL_events(trialwise_TTLs, save=(True, output_dir))
    
    aligned_trialwise_TTLs, recorded_start_ts = remove_laser_trials(aligned_trialwise_TTLs, recorded_start_ts)
    
    assert aligned_trialwise_TTLs[0]['start_time'] == recorded_start_ts[0]
    recorded_start_ts = np.array(recorded_start_ts)
    print('Done.')
    # --------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------- #
    # Trial start in absolute time from the behavior control system
    # --------------------------------------------------------------------- #
    # Load trial events structure
    session_data = loadmat(behavior_mat_file, simplify_cells=True)['SessionData']
    
    n_trials = session_data['nTrials']
    behav_start_ts = session_data['TrialStartTimestamp']
    # --------------------------------------------------------------------- #
    

    # --------------------------------------------------------------------- #
    # Reconcile the recorded and behavioral timestamps
    # --------------------------------------------------------------------- #
    print('Reconciling recorded and behavioral timestamps...')
    # First check the number of trials
    print(f'{n_trials} behavioral trials, {len(aligned_trialwise_TTLs)} TTLs')
    
    # Match timestamps - in case of mismatch, try to fix
    if not is_match(behav_start_ts, recorded_start_ts):
        print("Timestamps do not match.  Removing ISI violations...")
        # note: obsolete due the introduction of TTL parsing
        recorded_start_ts = clear_ttls_with_isi_violation(recorded_start_ts) 
        
        if not is_match(behav_start_ts, recorded_start_ts):
            print('Still no match. Try to match time series by shifting...')
            recorded_start_ts = reconcile_with_shift(behav_start_ts, recorded_start_ts)  
            
            if not is_match(behav_start_ts, recorded_start_ts):
                print('Still no match. Try to interpolate missing TTLs...')
                recorded_start_ts = try_interpolation(behav_start_ts, recorded_start_ts, 
                                                      first_trial_of_next_session=n_trials+1) 
                
                if not is_match(behav_start_ts, recorded_start_ts):
                    Exception('Matching TTLs failed.')

    print('Timestamp matching resolved.')

    
    # If the timestamp arrays have different lengths, eliminate timestamps
    # from the longer series to make them the same length
    if len(recorded_start_ts) > len(behav_start_ts):
        # missing timestamp in behavior file (likely reason: autosave was used)
        recorded_start_ts = recorded_start_ts[:len(behav_start_ts)]
    elif len(recorded_start_ts) < len(behav_start_ts):  
        # missing timestamp from recording sys (likely reason: recording stopped)
        session_data = shorten_session_data(session_data, len(recorded_start_ts))
        print('Trial Event File shortened to match TTL!')
    print('Done.')

    # --------------------------------------------------------------------- #
    # Finally, save the trial-start timestamps of the aligned, recorded TTLs
    # These will be used to align the trialwise spiking data
    # --------------------------------------------------------------------- #
    
    session_data['TrialStartAligned'] = recorded_start_ts
    
    dump(session_data, output_dir + 'TrialEvents.npy', compress=3)
    print('Results saved to ' + output_dir + 'TrialEvents.npy.')



def is_match(s1, s2):
    """
    Check if two time series match notwithstanding a constant drift.
    
    Note: abs(max()) is OK, because the derivative is usually a small neg. 
    number due to drift of the timestamps.  In contrast, max(abs)) would 
    require a higher tolerance taking the drift into account (if 2 event time 
    stamps are far, the drift between them can be large)
    
    return [bool] 
    """

    clen = min(len(s1), len(s2))
    
    # The difference between the timestamps on 2 systems may have a constant 
    # drift, but it's derivative should still be ~0
    max_difference = np.diff(s1[:clen] - s2[:clen]).max()
    return abs(max_difference) < 0.1 


def clear_ttls_with_isi_violation(ttl_signal, min_ISI=0.5):
    """
    Eliminate recorded TTL's within 0.5s from each other (broken TTL pulse).
    """        
    isi_violations = np.where(np.diff(ttl_signal) < min_ISI)[0]
    # ISI between index i and i+1 is difference index i, but we want to remove
    # index i+1, so add 1 to the ISI violation index
    isi_violations = (isi_violations + 1).astype('int')

    return np.array([t for i,t in enumerate(ttl_signal) if i not in isi_violations])


def reconcile_with_shift(s1, s2, compare_window=15):
    """
    Find the shift which minimizes the difference between two time series
    within a comparison window of fixed size.
    
    Returns the second time series, s2, shifted by the optimal shift amount.
    """
    room_to_shift = len(s2) - compare_window
    max_diffs = np.empty(room_to_shift)
    for shift_size in range(room_to_shift):
        # calculate difference in the function of shift
        s2_shift = s2[shift_size:(shift_size+compare_window)]
        max_diffs[shift_size] = max(np.diff(s1[:compare_window] - s2_shift))  

    # minimal difference = optimal shift
    smallest_discrepancy_idx = np.argmin(np.abs(max_diffs)) 
    adjusted_end = min(smallest_discrepancy_idx + len(s1), len(s2))
    return s2[smallest_discrepancy_idx:adjusted_end]
    
    
def try_interpolation(s1, s2, first_trial_of_next_session, attempts=10):
    """
    Interpolate missing TTLs or 
    delete superfluous TTLs (up to 10 errors)
    """
    for k in range(attempts):
        if not is_match(s1,s2):
            
            # make both series the same length and check if the difference
            # in the two series remains constant or varies
            min_end = min(len(s1), len(s2))
            d_diff_dt = np.diff(s1[:min_end]-s2[:min_end])
            
            # find the first problematic index, where the difference
            # between the two series changes drastically (>0.1)
            bad_idx = np.where(np.abs(d_diff_dt) > 0.1)[0] + 1
            
            if bad_idx == first_trial_of_next_session:
                print('Problem is from concatenated sessions!')

            if d_diff_dt[bad_idx - 1] < 0:  
                # interpolate only if the recorded timestamp came after the
                # controlled timestamp
                s3 = s2 - s2[0] + s1[0]  # shift s2 times in ref. to s1
                
                # average the difference in the two signals at the two indices
                # around bad_idx, and subtract that amount from s1 at bad_idx
                ave_diff = 0.5 * ((s1[bad_idx+1] - s3[bad_idx+1]) + 
                                  (s1[bad_idx-1] - s3[bad_idx-1]))
                correction = s1[bad_idx] - ave_diff
                
                # return to s2 reference timeframe and save the interpolated
                # value in the bad index
                s2[bad_idx] = correction + s2[0] - s1[0] 
            else: 
                # if the recorded timestamp somehow came after the control
                # (which is not possible), there was some unreconcilable error,
                # in which case the data point should be deleted
                del s2[bad_idx]
    return s2


def shorten_session_data(session_data, n_trials):
    """
    Eliminate behavioral trials that do not have corresponding TTL recordings
    (HACK TO WORK FOR DUAL2AFC).
    The only two fields really affected by this (the way it's written)
    are TrialStartTimestamp and arrays in the Custom field.
    author: Torben Ott (2019)
    """
    for obj_key in session_data.keys():
        if type(session_data[obj_key]) == np.ndarray:
            if len(session_data[obj_key]) >= n_trials:
                session_data[obj_key] = session_data[obj_key][:n_trials]

    for obj_key in session_data['Custom'].keys():
        if  type(session_data['Custom'][obj_key]) == np.ndarray:
            if len(session_data['Custom'][obj_key]) >= n_trials:
                session_data['Custom'][obj_key] = session_data['Custom'][obj_key][:n_trials]
    
    session_data['nTrials'] = n_trials

    return session_data



    
def calc_event_outcomes(output_dir):
    """
    Creates additional useful fields in the session data (trial events).

    :param output_dir: [string] directory where TrialEvents.npy was saved.
    :return: [None] Overwrites TrialEvents.npy
    """
    # Load the session data in TrialEvents.npy
    _sd = load(output_dir + 'TrialEvents.npy')

    WT_low_threshold = 0.  # Lower cut-off for waiting time turning all to NaN
    
    n_trials = _sd['nTrials'] - 1  # throw out last trial (may be incomplete)
    _sd['nTrials'] = n_trials
    _sd['TrialNumber'] = np.arange(n_trials)

    def one_zero_idx(data_obj):
        one_choice_idx = data_obj == 1.
        zero_choice_idx = data_obj == 0.        
        return one_choice_idx, zero_choice_idx
        
    # Chosen direction (1=left, 2=right, -1=nan)
    choice_left = _sd['Custom']['ChoiceLeft'][:n_trials]
    left_choice_idx, right_choice_idx = one_zero_idx(choice_left)
    _sd['ChosenDirection'] = -1 * np.ones(n_trials)
    _sd['ChosenDirection'][left_choice_idx] = 1
    _sd['ChosenDirection'][right_choice_idx] = 2

    # Correct and error trials
    choice_correct = _sd['Custom']['ChoiceCorrect'][:n_trials]
    _sd['CorrectChoice'], _sd['PunishedTrial'] = one_zero_idx(choice_correct)

    # Trial where rat gave a response
    _sd['CompletedTrial'] = (choice_left > -1) & (_sd['TrialNumber'] > 30)

    # Rewarded Trials
    _sd['Rewarded'], no_reward = one_zero_idx(_sd['Custom']['Rewarded'][:n_trials])

    # Trials where rat sampled but did not respond
    complete, incomplete = one_zero_idx(_sd['CompletedTrial'])
    early_withdrawal = _sd['Custom']['EarlyWithdrawal'][:n_trials] == 1
    _sd['UnansweredTrials'] = incomplete & early_withdrawal
    assert _sd['UnansweredTrials'].sum() <= min(early_withdrawal.sum(), incomplete.sum())

    # CatchTrial
    _sd['CatchTrial'] = _sd['Custom']['CatchTrial'][:n_trials]

    # Correct catch trials
    _sd['CompletedCatchTrial'] = complete & (_sd['CatchTrial']==1)

    # Correct trials, but rat was waiting too short
    wait_too_short = _sd['Custom']['FeedbackTime'][:n_trials] < 0.5
    _sd['CorrectShortWTTrial'] = _sd['CorrectChoice'] & wait_too_short

    # These are all the waiting time trials (correct catch and incorrect trials)
    catch_or_wrong = _sd['CompletedCatchTrial'] & _sd['PunishedTrial']
    _sd['CompletedWTTrial'] = catch_or_wrong & complete
    assert _sd['CompletedWTTrial'].sum() <= min(catch_or_wrong.sum(), complete.sum())
    
    # Trials where rat answered but did not receive reward
    punish_or_catch = _sd['PunishedTrial'] | _sd['CatchTrial']
    _sd['WaitingTimeTrial'] = punish_or_catch & complete
    
    # Waiting Time
    _sd['WaitingTime'] = _sd['Custom']['FeedbackTime']
    
    # Threshold for waiting time (e.g., negative waiting times are impossible)
    _sd['WaitingTime'][_sd['WaitingTime'] < WT_low_threshold] = np.nan
    
    # Modality
    _sd['Modality'] = 2 * np.ones(n_trials)
    _sd['SideReward'] = -1 * np.ones(n_trials)
    _sd['CompletedChosenDirection'] = -1 * np.ones(n_trials)
    _sd['ModReward'] = -1 * np.ones(n_trials)

    # Conditioning the trials
    for nt in range(n_trials):  
        """
        Defining trial types
        Defining DecisionType
            0 = Non-completed trials
            1 = Correct given click and not rewarded (catch trials consisting
                of real catch trials and trials that are statistically 
                incorrect but correct given click, later ones are most likely 
                50/50 trials)
            2 = Correct given click and rewarded
            3 = Incorrect given click and not rewarded
        """
        nt_reward = _sd['Rewarded'][nt]
        nt_mod = _sd['Modality'][nt]
        nt_complete = _sd['CompletedTrial'][nt]
        nt_chosen_dir = _sd['ChosenDirection'][nt]
        if nt_reward and nt_mod==1:
            code = 1
        elif nt_reward and nt_mod==2:
            code = 2
        elif ~nt_reward and nt_complete and nt_mod==1:
            code = 3
        elif ~nt_reward and nt_complete and nt_mod==2:
            code = 4
        else:
            code = np.nan
        _sd['SideReward'][nt] = code
        
        # Defining ChosenDirection (1 = Left, 2 = Right)
        if nt_complete and nt_chosen_dir > 0:
            _sd['CompletedChosenDirection'][nt] = nt_chosen_dir

        """
        Defining SideDecisionType
          1 = Left catch trials
          2 = Right catch trials
          3 = Left correct trials
          4 = Right correct trials
          5 = Incorrect left trials
          6 = Incorrect right trials
          7 = all remaining trials
        """

        if nt_mod==1 and nt_chosen_dir==1 and nt_complete:
            code2 = 1
        elif nt_mod==2 and nt_chosen_dir==1 and nt_complete:
            code2 = 2
        elif nt_mod==1 and nt_chosen_dir==2 and nt_complete:
            code2 = 3
        elif nt_mod==2 and nt_chosen_dir==2 and nt_complete:
            code2 = 4
        else:
            code2 = np.nan
        _sd['ModReward'][nt] = code2

    
    # Waiting time split
    _sd['WaitingTimeSplit'] = np.full(_sd['ChosenDirection'].shape, np.nan)
    
    complete_no_reward = complete & no_reward

    long_idx = _sd['WaitingTime'] >= 6.5
    midlong_idx = (_sd['WaitingTime']<6.5) & (_sd['WaitingTime']>=5.5)
    midshort_idx = (_sd['WaitingTime']<5.5) & (_sd['WaitingTime']>=4)
    short_idx = (_sd['WaitingTime']<4) & (_sd['WaitingTime']>=2.5)
    assert ~np.any(np.where(long_idx)[0] == np.where(midlong_idx)[0])
    assert ~np.any(np.where(midlong_idx)[0] == np.where(midshort_idx)[0])
    assert ~np.any(np.where(midshort_idx)[0] == np.where(short_idx)[0])

    _sd['WaitingTimeSplit'][complete_no_reward & short_idx] = 1
    _sd['WaitingTimeSplit'][complete_no_reward & midshort_idx] = 2
    _sd['WaitingTimeSplit'][complete_no_reward & midlong_idx] = 3
    _sd['WaitingTimeSplit'][complete_no_reward & long_idx] = 4
    

    # Defining ResponseOnset, ResponseStart and ResponseEnd
    _sd['StimulusOnset'] = np.zeros(n_trials)
    _sd['PokeCenterStart'] = np.zeros(n_trials)

    _sd['ResponseStart'] = np.zeros(n_trials)
    _sd['ResponseEnd'] = np.zeros(n_trials)

    _sd['LaserTrialTrainLength'] = np.zeros(n_trials)
    
    for nt in range(n_trials):
        nt_states = _sd['RawEvents']['Trial'][nt]['States']

        _sd['StimulusOnset'][nt] = nt_states['stimulus_delivery_min'][0]
        _sd['PokeCenterStart'][nt] = nt_states['stay_Cin'][0]

        if ~np.isnan(nt_states['start_Rin'][0]):
            _sd['ResponseStart'][nt] = nt_states['start_Rin'][0]
            _sd['ResponseEnd'][nt] = nt_states['start_Rin'][0] + _sd['Custom']['FeedbackTime'][nt]
        elif ~np.isnan(nt_states['start_Lin'][0]):
            _sd['ResponseStart'][nt] = nt_states['start_Lin'][0]
            _sd['ResponseEnd'][nt] = nt_states['start_Lin'][0] + _sd['Custom']['FeedbackTime'][nt]
        else:
            _sd['ResponseStart'][nt] = np.nan
            _sd['ResponseEnd'][nt] = np.nan

        nt_GUI = _sd['TrialSettings'][nt]['GUI']
        if 'LaserTrials' in nt_GUI.keys():
            if nt_GUI['LaserTrials'] > 0:
                if 'LaserTrainDuration_ms' in nt_GUI.keys():
                    _sd['LaserTrialTrainLength'][nt] = nt_GUI['LaserTrainDuration_ms']
                else: #old version
                    _sd['LaserTrialTrainLength'][nt] = np.nan
            else:
                _sd['LaserTrialTrainLength'][nt] = np.nan
        else: # not even Laser Trials settings, very old version
            _sd['LaserTrialTrainLength'][nt] = np.nan

    _sd['SamplingDuration'] = _sd['Custom']['ST'][:n_trials]
    _sd['StimulusOffset'] = _sd['StimulusOnset'] + _sd['SamplingDuration']
    
    _sd['ChosenDirectionBis'] = _sd['ChosenDirection']
    _sd['ChosenDirectionBis'][_sd['ChosenDirectionBis']==3] = np.nan
    
    # Correct length of TrialStartAligned
    _sd['TrialStartAligned'] = _sd['TrialStartAligned'][:n_trials]
    _sd['TrialStartTimestamp'] = _sd['TrialStartTimestamp'][:n_trials]
    _sd['TrialSettings'] = _sd['TrialSettings'][:n_trials]
    
    #laser trials
    if 'LaserTrial' in _sd['Custom'].keys() and _sd['Custom']['LaserTrial'].sum() > 0:
        if 'LaserTrialTrainStart' in _sd['Custom'].keys():
            _sd['LaserTrialTrainStart'] = _sd['Custom']['LaserTrialTrainStart'][:n_trials]
            _sd['LaserTrialTrainStartAbs'] = _sd['LaserTrialTrainStart'] + _sd['ResponseStart']
            _sd['LaserTrial'] = _sd['Custom']['LaserTrial'][:n_trials]
            _sd['LaserTrial'][_sd['CompletedTrial']==0] = 0
            _sd['LaserTrial'][_sd['LaserTrialTrainStartAbs'] >_sd['ResponseEnd']] = 0
            _sd['LaserTrialTrainStartAbs'][_sd['LaserTrial'] !=1 ] = np.nan
            _sd['LaserTrialTrainStart'][_sd['LaserTrial'] !=1 ] = np.nan

            _sd['CompletedWTLaserTrial'] = _sd['LaserTrial']
            _sd['CompletedWTLaserTrial'][_sd['CompletedWTTrial'] !=1 ] = np.nan
        else: #old version, laser during entire time investment
            _sd['LaserTrialTrainStart'] = np.zeros(n_trials)
            _sd['LaserTrialTrainStartAbs'] = _sd['ResponseStart']
            _sd['LaserTrial'] = _sd['Custom']['LaserTrial'][:n_trials]
            _sd['LaserTrial'][_sd['CompletedTrial']==0] = 0
            _sd['LaserTrialTrainStartAbs'][_sd['LaserTrial'] !=1 ] = np.nan
            _sd['LaserTrialTrainStart'][_sd['LaserTrial'] !=1 ] = np.nan
    else:
        _sd['LaserTrialTrainStart'] = np.full(n_trials, np.nan)
        _sd['LaserTrialTrainStartAbs'] = np.full(n_trials, np.nan)
        _sd['LaserTrial'] = np.zeros(n_trials)
        _sd['CompletedWTLaserTrial'] = np.full(n_trials, np.nan)
        _sd['CompletedWTLaserTrial'][_sd['CompletedWTTrial'] ==1 ] = 0

    if 'BlockNumber' in _sd['Custom'].keys():
        _sd['BlockNumber'] = _sd['Custom']['BlockNumber'][:n_trials]

    if 'RewardMagnitude' in _sd['Custom'].keys():
        _sd['RelativeReward'] = _sd['Custom']['RewardMagnitude'][:, 0] - _sd['Custom']['RewardMagnitude'][:, 1]
        _sd['RelativeReward'] = _sd['RelativeReward'][:n_trials]

    # discrimination measures
    _sd['MostClickSide'] = -1 * np.ones(n_trials)
    if 'AuditoryOmega' in _sd['Custom'].keys():
        _sd['OmegaDiscri'] = 2 * np.abs(_sd['Custom']['AuditoryOmega'][:n_trials] - 0.5)
        _sd['NRightClicks'] = np.zeros(n_trials)
        _sd['NLeftClicks'] = np.zeros(n_trials)
        for trial_i in range(n_trials):
            rct = _sd['Custom']['RightClickTrain'][trial_i]
            if type(rct) == np.ndarray:
                _sd['NRightClicks'][trial_i] = len(rct)
            else:
                _sd['NRightClicks'][trial_i] = 1

            lct = _sd['Custom']['LeftClickTrain'][trial_i]
            if type(lct) == np.ndarray:
                _sd['NLeftClicks'][trial_i] = len(lct)
            else:
                _sd['NLeftClicks'][trial_i] = 1
        _sd['RatioDiscri'] = np.log10(_sd['NRightClicks'] / _sd['NLeftClicks'])
        _sd['BetaDiscri'] = (_sd['NRightClicks'] - _sd['NLeftClicks']) / (_sd['NRightClicks'] + _sd['NLeftClicks'])
        _sd['AbsBetaDiscri'] = np.abs(_sd['BetaDiscri'])
        _sd['AbsRatioDiscri'] = np.abs(_sd['RatioDiscri'])

        _sd['MostClickSide'][_sd['NRightClicks'] > _sd['NLeftClicks']] = 2
        _sd['MostClickSide'][_sd['NRightClicks'] < _sd['NLeftClicks']] = 1
        _sd['MostClickSide'][_sd['NRightClicks'] == _sd['NLeftClicks']] = 3
        _sd['ChoiceGivenClick'] = _sd['MostClickSide'] == _sd['ChosenDirection']
    else:
        _sd['ChoiceGivenClick'] = _sd['CorrectChoice'][:n_trials]

    _sd['DV'] = _sd['Custom']['DV'][:n_trials]


    if 'RewardMagnitude' in _sd['Custom'].keys():
        _sd['RewardMagnitude'] = _sd['Custom']['RewardMagnitude'][:n_trials, :]

        # Conditioning the trials
        _sd['RewardMagnitudeCorrect'] = -1 * np.ones(n_trials)
        for nt in range(n_trials):
            rewMagTrial = _sd['RewardMagnitude'][nt, :]

            # chosen direction
            cd = int(_sd['ChosenDirection'][nt])

            if cd == 3:
                _sd['RewardMagnitudeCorrect'][nt] = np.nan
            else:
                _sd['RewardMagnitudeCorrect'][nt] = rewMagTrial[cd - 1]

    dump(_sd, output_dir + 'TrialEvents.npy', compress=3)

    print('Trial event conditions/outcomes calculated and added to TrialEvents.npy.')


def create_behavioral_dataframe(output_dir):
    """
    Create simpler copy of Trial Events to turn into pandas dataframe
    """
    # Load the session data in TrialEvents.npy
    _sd = load(output_dir + 'TrialEvents.npy')

    # Remove keys no longer needed for spiking data alignment
    n_keys_start = len(_sd.keys())

    _sd.pop('Custom')
    _sd.pop('RawEvents')
    _sd.pop('nTrials')
    _sd.pop('TrialSettings')
    _sd.pop('Settings')

    if 'RawData' in _sd.keys():
        _sd.pop('RawData')

    if 'RewardMagnitude' in _sd.keys():
        _sd.pop('RewardMagnitude')

    try:
        _sd.pop('CompletedChosenDirection')
        _sd.pop('CompletedCatchTrial')

        _sd.pop('ModReward')
        _sd.pop('SideReward')
        _sd.pop('ChosenDirectionBis')
        _sd.pop('CorrectShortWTTrial')

        _sd.pop('CompletedWTLaserTrial')
        _sd.pop('CompletedWTTrial')
        _sd.pop('PunishedTrial')
    except:
        print('Some fields were not removed')

    print(f"{n_keys_start - len(_sd.keys())} fields removed.")

    # Convert to dataframe and save
    behav_dict = {key: np.array(_sd[key]).squeeze() for key in _sd if '__' not in key and key!='Settings'}
    behav_df = pd.DataFrame.from_dict(behav_dict)

    # so that we can use this code for session data that doesnt have catch trials!
    if 'CatchTrial' in behav_df.keys():
        behav_df.CatchTrial = behav_df.CatchTrial.astype('bool')
    if 'CompletedTrial' in behav_df.keys():
        behav_df.CompletedTrial = behav_df.CompletedTrial.astype('bool')

    dump(behav_df, output_dir + "behav_df", compress=3)

    print('Bahvaioral dataframe saved to: ' + output_dir + "behav_df")


def align_trialwise_spike_times_to_start(metadata, datapath, downsample_dt, TOY_DATA):
    if TOY_DATA:
        spike_mat = load(datapath + "toy_spikes.npy")
        behav_df = load(datapath + 'toy_behav_df')
    else:
        # load neural data: [number of neurons x time bins in ms]
        spike_mat = load(datapath + "spike_mat_in_ms.npy")['spike_mat']

        # make pandas behavior dataframe
        behav_df = load(datapath + 'behav_df')

    # format entries of dataframe for analysis (e.g., int->bool)
    cbehav_df = bu.convert_df(behav_df,
                              session_type="SessionData",
                              WTThresh=metadata['time_investment'],
                              trim_last_trial=~TOY_DATA)


    # align spike times to behavioral data timeframe
    # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
    trialwise_spike_mat_start_aligned, _ = trace_utils.trial_start_align(cbehav_df, spike_mat, sps=1000)

    # subsample (bin) data:
    # [n_neurons x n_trials x (-1 means numpy calculates: trial_len / dt) x ds]
    # then sum over the dt bins
    n_neurons = trialwise_spike_mat_start_aligned.shape[0]
    n_trials = trialwise_spike_mat_start_aligned.shape[1]
    trial_binned_mat_start_align = trialwise_spike_mat_start_aligned.reshape(n_neurons, n_trials, -1, downsample_dt)
    trial_binned_mat_start_align = trial_binned_mat_start_align.sum(axis=-1)  # sum over bins

    results = {'binned_mat': trial_binned_mat_start_align, 'downsample_dt': downsample_dt}
    dump(results, datapath + 'trial_binned_mat_start_align.npy', compress=3)

    return trial_binned_mat_start_align, cbehav_df