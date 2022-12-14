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
from scipy.io import loadmat

from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 \
    import get_Trodes_timestamps, readTrodesExtractedDataFile


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
    
    Saves
    -------
    TTL codes as time series with accompanying timestamps in 'TTL_events.npy'
    in the directory save_dir
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


def make_trial_events(cellbase_dir, behavior_mat_file):
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
    
    # --------------------------------------------------------------------- #
    # Trial start time recorded by the recording system (Neuralynx)
    # --------------------------------------------------------------------- #
    # Load converted TRODES event file 
    try:
        TTL_results = load(cellbase_dir + 'TTL_events.npy')
    except:
        print('TTL_events.npy file not found.  Make sure the TTL events \n\
        have been extraced from the TRODES .DIO files.')

    trialwise_TTLs = group_codes_and_timestamps_by_trial(**TTL_results)
    
    aligned_trialwise_TTLs, recorded_start_ts = align_TTL_events(trialwise_TTLs, save=(True, cellbase_dir))
    
    aligned_trialwise_TTLs, recorded_start_ts = remove_laser_trials(aligned_trialwise_TTLs, recorded_start_ts)
    
    assert aligned_trialwise_TTLs[0]['start_time'] == recorded_start_ts[0]  
    # --------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------- #
    # Trial start in absolute time from the behavior control system
    # --------------------------------------------------------------------- #
    # Load trial events structure
    session_data = loadmat(cellbase_dir + behavior_mat_file)['SessionData']
    
    # change_ind = TE.TE.nTrials(1) + 1
    n_trials = session_data[0,0]['nTrials'][0][0] + 1
    behav_start_ts = session_data[0,0]['TrialStartTimestamp'][0]
    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # Reconcile the recorded and behavioral timestamps
    # --------------------------------------------------------------------- #
    
    # # Match timestamps - in case of mismatch, try to fix
    # if ~ismatch(ts,son2)
    #     # note: obsolete due the introduction of TTL parsing
    #     son2 = clearttls(son2) # eliminate recorded TTL's within 0.5s from each other - broken TTL pulse
    #     if ~ismatch(ts,son2)
    #         son2 = trytomatch(ts,son2)  # try to match time series by shifting
    #         if ~ismatch(ts,son2)
    #             son2 = tryinterp(ts,son2, change_ind) # interpolate missing TTL's or delete superfluous TTL's up to 10 erroneous TTl's
    #             if ~ismatch(ts,son2)  # TTL matching failure
    #                 error('MakeTrialEvents:TTLmatch','Matching TTLs failed.')
    #             else
    #                 warning('MakeTrialEvents:TTLmatch','Missing TTL interpolated.')

    #         else
    #             warning('MakeTrialEvents:TTLmatch','Shifted TTL series.')

    #     else
    #         warning('MakeTrialEvents:TTLmatch','Broken TTLs cleared.')

    
    # # Eliminate last TTL's recorded in only one system
    # sto = TE2.TrialStartTimestamp
    # if length(son2) > length(ts):   # time not saved in behavior file (likely reason: autosave was used)
    #     son2 = son2(1:length(ts))
    # elif length(son2) < length(ts):  # time not recorded on Neuralynx (likely reason: recording stopped)
    #     shinx = 1:length(son2)
    #     ts = ts(shinx)
    #     sto = sto(shinx)
    #     TE2 = shortenTE(TE2,shinx)
    #     warning('Trial Event File shortened to match TTL!')

    
    # TE2.TrialStartAligned = son2
    
    
    # # Save synchronized 'TrialEvents' file
    # save([sessionpath filesep 'TrialEvents.mat'],'-struct','TE2')
    
    # if ~isempty(TE2.TrialStartTimestamp),
    #     save([sessionpath filesep 'TrialEvents.mat'],'-struct','TE2')
    # else
    #     error('MakeTrialEvents:noOutput','Synchronization process failed.')

    
    # if ~isempty(Alignedtrialwise_TTLsAll),
    #     save([sessionpath filesep 'Alignedtrialwise_TTLsAll.mat'],'Alignedtrialwise_TTLsAll')
    

    # if ~isempty(Alignedtrialwise_TTLs),
    #     save([sessionpath filesep 'Alignedtrialwise_TTLs.mat'],'Alignedtrialwise_TTLsAll')



# def ismatch(ts, son2):
#     """
#     Check if the two time series match notwithstanding a constant drift.
    
#     Note: abs(max()) is OK, because the derivative is usually a small neg. 
#     number due to drift of the timestamps.  In contrast, max(abs)) would 
#     require a higher tolerance taking the drift into account (if 2 event time 
#     stamps are far, the drift between them can be large)
    
#     return [bool] 
#     """

#     clen = min(len(ts), len(son2))
    
#     # The difference between the timestamps on 2 systems may have a constant 
#     # drift, but it's derivative should still be ~0
#     max_difference = np.diff(ts[:(clen - 1)] - son2[:(clen - 1)]).max()
#     return abs(max_difference) < 0.1 


# def tryinterp(ts, son2, change_point):
    
#     # Interpolate missing TTL's or delete superfluous TTL's up to 10 erroneous TTl's
#     for k = 1:10
#         if ~ismatch(ts,son2)
#             son3 = son2 - son2(1) + ts(1)
#             adt = diff(ts(1:min(length(ts),length(son2)))-son2(1:min(length(ts),length(son2))))
#             badinx = find(abs(adt)>0.1,1,'first') + 1# find problematic index
            
#             if badinx == change_point
#                     fprintf('problem is from concatenated sessions')
#             end
#             if adt(badinx-1) < 0    # interploate
#                 ins = ts(badinx) - linterp([ts(badinx-1) ts(badinx+1)],[ts(badinx-1)-son3(badinx-1) ts(badinx+1)-son3(badinx)],ts(badinx))
#                 son2 = [son2(1:badinx-1) ins+son2(1)-ts(1) son2(badinx:end)]
#             else
#                 son2(badinx) = []   # delete
#     return son2
    

# def trytomatch(ts,son2):
    
#     # Try to match time series by shifting
#     len = length(son2) - 15
#     minx = nan(1,len)
#     for k = 1:len
#         minx(k) = max(diff(ts(1:15)-son2(k:k+14)))  # calculate difference in the function of shift

#     mn = min(abs(minx))
#     minx2 = find(abs(minx)==mn)
#     minx2 = minx2(1)   # find minimal difference = optimal shift
#     son2 = son2(minx2:min(minx2+length(ts)-1,length(son2)))
    
#     return son2
    

# def clearttls(son2):
    
#     # Eliminate recorded TTL's within 0.5s from each other
#     inx = []
#     for k = 1:length(son2)-1
#         s1 = son2(k)
#         s2 = son2(k+1)
#         if s2 - s1 < 0.5
#             inx = [inx k+1] ##ok<AGROW>

#     son2(inx) = []
#     return son2


# def shortenTE(TE2,shinx):
#     ###HACK TO WORK FOR DUAL2AFC 
#     #TO 2019
#     # Eliminate behavioral trials
#     fnm = fieldnames(TE2)
#     for k = 1:length(fnm)
#         if length(TE2.(fnm{k}))>=shinx(end)
#         TE2.(fnm{k}) = TE2.(fnm{k})(shinx)

    
#     fnm = fieldnames(TE2.Custom)
#     for k = 1:length(fnm):
#         if length(TE2.Custom.(fnm{k}))>=shinx(end)
#         TE2.Custom.(fnm{k}) = TE2.Custom.(fnm{k})(shinx)
    
#     TE2.nTrials=length(shinx)+1

#     return TE2



    
def create_behavioral_dataframe(behavior_mat_file):
    """
    Requires TrialEvents.mat from make_trial_events().
    """
    # Load the behavioral file
    # Uses a string that is specific to the behaviour file name (depends on task)
    print('Loading behavior file: ' + behavior_mat_file)
    
        
    #behav_mat = loadmat(behavior_mat_file)
    
    #session_trial_event_data = behav_mat['SessionData']
    # TEbis=SessionData
    # save(fullfile(Directory,'TE.mat'),'TE')
    
    
    
    """
    ## Create additional fields in the Trial Event structure
    disp('Creating additional fields in the Trial Event structure')
    
    TEbis=load(fullfile(Directory,'TrialEvents.mat'))
    TE=TEbis
    
    WT_low_threshold=0 # Lower cut off for waiting time turning all to NaN
    
    #nTrial
    nTrials = TE.nTrials-1
    TEbis.nTrials=nTrials
    TEbis.TrialNumber = 1:nTrials
    #discrimination measures
    #TEbis.OmegaDiscri=2*abs(TE.Custom.AuditoryOmega(1:nTrials)-0.5)
    #TEbis.NRightClicks = cellfun(@length,TE.Custom.RightClickTrain(1:nTrials))
    #TEbis.NLeftClicks = cellfun(@length,TE.Custom.LeftClickTrain(1:nTrials))
    #TEbis.RatioDiscri=log10(TEbis.NRightClicks./TEbis.NLeftClicks)
    #TEbis.BetaDiscri=(TEbis.NRightClicks-TEbis.NLeftClicks)./(TEbis.NRightClicks+TEbis.NLeftClicks)
    #TEbis.AbsBetaDiscri=abs(TEbis.BetaDiscri)
    #TEbis.AbsRatioDiscri=abs(TEbis.RatioDiscri)
    #chosen direction
    TEbis.ChosenDirection = 3*ones(1,nTrials)
    TEbis.ChosenDirection(TE.Custom.ChoiceLeft==1)=1#1=left 2=right
    TEbis.ChosenDirection(TE.Custom.ChoiceLeft==0)=2
    # Correct and error trials
    TEbis.CorrectChoice=TE.Custom.ChoiceCorrect(1:nTrials)
    TEbis.PunishedTrial=TE.Custom.ChoiceCorrect(1:nTrials)==0
    #most click side
    #TEbis.MostClickSide(TEbis.NRightClicks>TEbis.NLeftClicks) = 2
    #TEbis.MostClickSide(TEbis.NRightClicks<TEbis.NLeftClicks) = 1
    #TEbis.MostClickSide(TEbis.NRightClicks==TEbis.NLeftClicks)  = 3
    # Trial where rat gave a response
    TEbis.CompletedTrial= ~isnan(TE.Custom.ChoiceLeft(1:nTrials)) & TEbis.TrialNumber>30
    # Rewarded Trials
    TEbis.Rewarded=TE.Custom.Rewarded(1:nTrials)==1
    # Trials where rat sampled but did not respond
    TEbis.UnansweredTrials=(TEbis.CompletedTrial(1:nTrials)==0 & TE.Custom.EarlyWithdrawal(1:nTrials)==1)
    #CatchTrial
    TEbis.CatchTrial = TE.Custom.CatchTrial(1:nTrials)
    # Correct catch trials
    TEbis.CompletedCatchTrial=TEbis.CompletedTrial(1:nTrials)==1 & TE.Custom.CatchTrial(1:nTrials)==1 
    # Correct trials, but rat was waiting too short
    TEbis.CorrectShortWTTrial=TE.Custom.ChoiceCorrect(1:nTrials)==1 & TE.Custom.FeedbackTime(1:nTrials)<0.5
    # These are all the waiting time trials (correct catch and incorrect trials)
    TEbis.CompletedWTTrial= (TEbis.CompletedCatchTrial(1:nTrials)==1 | TEbis.PunishedTrial(1:nTrials)==1) & TEbis.CompletedTrial(1:nTrials)
    
    # Trials were rat answered but did not receive reward
    WTTrial=TEbis.CompletedTrial(1:nTrials)==1 & (TEbis.PunishedTrial(1:nTrials)==1 | TE.Custom.CatchTrial(1:nTrials)==1)
    
    TEbis.WaitingTimeTrial=WTTrial
    
    # Waiting Time
    TEbis.WaitingTime=TE.Custom.FeedbackTime
    
    # Threshold for waiting time
    TEbis.WaitingTime(TEbis.WaitingTime<WT_low_threshold)=NaN
    
    # This is to indicate whether choice matches actual click train (important for difficult trials)
    #TEbis.ChoiceGivenClick=TEbis.MostClickSide==TEbis.ChosenDirection
    #modality
    TEbis.Modality = 2*ones(1,nTrials)
    ## Conditioning the trials
    for nt=1:nTrials
        
        #{
        Defining trial types
        Defining DecisionType
            0 = Non-completed trials
            1 = Correct given click and not rewarded (catch trials consisting
                of real catch trials and trials that are statistically 
                incorrect but correct given click, later ones are most likely 
                50/50 trials)
            2 = Correct given click and rewarded
            3 = Incorrect given click and not rewarded
        #}
    
     #   if TEbis.CompletedTrial(nt)==0
     #       TEbis.DecisionType(nt)=NaN
     #   elseif (TEbis.CompletedCatchTrial(nt)==1 || (TEbis.Rewarded(nt)==0 && TEbis.CompletedTrial(nt)==1)) && TEbis.ChoiceGivenClick(nt)==1
     #       TEbis.DecisionType(nt)=1
     #   elseif TEbis.Rewarded(nt)==1
     #       TEbis.DecisionType(nt)=2
     #   elseif (TEbis.CompletedCatchTrial(nt)==1 || (TEbis.Rewarded(nt)==0 && TEbis.CompletedTrial(nt)==1)) && TEbis.ChoiceGivenClick(nt)==0
     #       TEbis.DecisionType(nt)=3
     #   end
        
        if TEbis.Rewarded(nt)==1 && TEbis.Modality(nt)==1
            TEbis.SideReward(nt)=1
        elseif TEbis.Rewarded(nt)==1  && TEbis.Modality(nt)==2
            TEbis.SideReward(nt)=2
        elseif TEbis.Rewarded(nt)==0 && TEbis.CompletedTrial(nt)==1  && TEbis.Modality(nt)==1
            TEbis.SideReward(nt)=3
        elseif TEbis.Rewarded(nt)==0 && TEbis.CompletedTrial(nt)==1  && TEbis.Modality(nt)==2
            TEbis.SideReward(nt)=4
        else
            TEbis.SideReward(nt)=NaN
        end
        
        # Defining ChosenDirection (1 = Left, 2 = Right)
        if TEbis.CompletedTrial(nt)==1 && TEbis.ChosenDirection(nt)==1
            TEbis.CompletedChosenDirection(nt)=1
        elseif TEbis.CompletedTrial(nt)==1 && TEbis.ChosenDirection(nt)==2
            TEbis.CompletedChosenDirection(nt)=2
        end
        
        #{
        Defining SideDecisionType
         1 = Left catch trials
         2 = Right catch trials
         3 = Left correct trials
         4 = Right correct trials
         5 = Incorrect left trials
         6 = Incorrect right trials
         7 = all remaining trials
        #}
        
    #    if TEbis.DecisionType(nt)==1 && TEbis.ChosenDirection(nt)==1
    #        TEbis.SideDecisionType(nt)=1
    #    elseif TEbis.DecisionType(nt)==1 && TEbis.ChosenDirection(nt)==2
    #        TEbis.SideDecisionType(nt)=2
    #    elseif TEbis.DecisionType(nt)==2 && TEbis.ChosenDirection(nt)==1
    #        TEbis.SideDecisionType(nt)=3
    #    elseif TEbis.DecisionType(nt)==2 && TEbis.ChosenDirection(nt)==2
    #        TEbis.SideDecisionType(nt)=4
    #    elseif TEbis.DecisionType(nt)==3 && TEbis.ChosenDirection(nt)==1
    #        TEbis.SideDecisionType(nt)=5
    #    elseif TEbis.DecisionType(nt)==3 && TEbis.ChosenDirection(nt)==2
    #        TEbis.SideDecisionType(nt)=6
    #    else
    #        TEbis.SideDecisionType(nt)=7
    #    end
        
        if TEbis.Modality(nt)==1 && TEbis.ChosenDirection(nt)==1 && TEbis.CompletedTrial(nt)==1
            TEbis.ModReward(nt)=1
        elseif TEbis.Modality(nt)==2 && TEbis.ChosenDirection(nt)==1 && TEbis.CompletedTrial(nt)==1
            TEbis.ModReward(nt)=2
        elseif TEbis.Modality(nt)==1 && TEbis.ChosenDirection(nt)==2 && TEbis.CompletedTrial(nt)==1
            TEbis.ModReward(nt)=3
        elseif TEbis.Modality(nt)==2 && TEbis.ChosenDirection(nt)==2 && TEbis.CompletedTrial(nt)==1
            TEbis.ModReward(nt)=4
        else
            TEbis.ModReward(nt)=NaN
        end
        
    end
    
    #waiting time split
    TEbis.WaitingTimeSplit=NaN(size(TEbis.ChosenDirection))
    
    Long=TEbis.CompletedTrial==1 & TEbis.Rewarded==0 & TEbis.WaitingTime>=6.5
    MidLong=TEbis.CompletedTrial==1 & TEbis.Rewarded==0 & TEbis.WaitingTime<6.5 & TEbis.WaitingTime>=5.5 
    MidShort=TEbis.CompletedTrial==1 & TEbis.Rewarded==0 & TEbis.WaitingTime<5.5 & TEbis.WaitingTime>=4
    Short=TEbis.CompletedTrial==1 & TEbis.Rewarded==0 & TEbis.WaitingTime<4 & TEbis.WaitingTime>=2.5
    
    TEbis.WaitingTimeSplit(Short)=1
    TEbis.WaitingTimeSplit(MidShort)=2
    TEbis.WaitingTimeSplit(MidLong)=3
    TEbis.WaitingTimeSplit(Long)=4
    
    
    ## Saving conditioned trials
    save(fullfile(Directory,'TEbis.mat'),'TEbis')
    save(fullfile(Directory,  'TrialEvents.mat'),'-struct','TEbis')
    
    
    ## Defining ResponseOnset, ResponseStart and ResponseEnd
    TEbis.ResponseStart=zeros(1,TEbis.nTrials)
    TEbis.ResponseEnd=zeros(1,TEbis.nTrials)
    TEbis.PokeCenterStart=zeros(1,TEbis.nTrials)
    TEbis.StimulusOnset=zeros(1,TEbis.nTrials)
     TEbis.LaserTrialTrainLength=zeros(1,TEbis.nTrials)
    
    for nt=1:TEbis.nTrials
        TEbis.StimulusOnset(nt)=TE.RawEvents.Trial{nt}.States.stimulus_delivery_min(1)
        TEbis.PokeCenterStart(nt)=TE.RawEvents.Trial{nt}.States.stay_Cin(1)
        if ~isnan(TE.RawEvents.Trial{nt}.States.start_Rin(1))
            TEbis.ResponseStart(nt)=TE.RawEvents.Trial{nt}.States.start_Rin(1)
            TEbis.ResponseEnd(nt)=TE.RawEvents.Trial{nt}.States.start_Rin(1) + TE.Custom.FeedbackTime(nt)
        elseif ~isnan(TE.RawEvents.Trial{nt}.States.start_Lin(1))
            TEbis.ResponseStart(nt)=TE.RawEvents.Trial{nt}.States.start_Lin(1)
            TEbis.ResponseEnd(nt)=TE.RawEvents.Trial{nt}.States.start_Lin(1) + TE.Custom.FeedbackTime(nt)
            #     elseif ~isnan(TE.RawEvents.Trial{nt}.States.PunishStart(1)) && isnan(TE.RawEvents.Trial{nt}.States.StillWaiting(end))
            #         TEbis.ResponseStart(nt)=TE.RawEvents.Trial{nt}.States.PunishStart(1)
            #         TEbis.ResponseEnd(nt)=(TE.RawEvents.Trial{nt}.States.Punish(end))
            #     elseif ~isnan(TE.RawEvents.Trial{nt}.States.PunishStart(1))
            #         TEbis.ResponseStart(nt)=TE.RawEvents.Trial{nt}.States.PunishStart(1)
            #         TEbis.ResponseEnd(nt)=(TE.RawEvents.Trial{nt}.States.StillWaiting(end))
        else
            TEbis.ResponseStart(nt)=NaN
            TEbis.ResponseEnd(nt)=NaN
        end
        if isfield(TE.TrialSettings(nt).GUI,'LaserTrials')
        if TE.TrialSettings(nt).GUI.LaserTrials>0
            if isfield(TE.TrialSettings(nt).GUI,'LaserTrainDuration_ms')
                TEbis.LaserTrialTrainLength(nt) = TE.TrialSettings(nt).GUI.LaserTrainDuration_ms
            else #old version
                TEbis.LaserTrialTrainLength(nt)=NaN 
            end
        end
        else #not even Laser Trials settings, very old version
            TEbis.LaserTrialTrainLength(nt)=NaN
        end
    end
    TEbis.SamplingDuration = TE.Custom.ST(1:nTrials)
    TEbis.StimulusOffset=TEbis.StimulusOnset+TEbis.SamplingDuration
    
    TEbis.ChosenDirectionBis=TEbis.ChosenDirection
    TEbis.ChosenDirectionBis(TEbis.ChosenDirectionBis==3)=NaN
    
    #correct length of TrialStartAligned
    TEbis.TrialStartAligned = TEbis.TrialStartAligned(1:TEbis.nTrials)
    TEbis.TrialStartTimestamp = TEbis.TrialStartTimestamp(1:TEbis.nTrials)
    TEbis.TrialSettings = TEbis.TrialSettings(1:TEbis.nTrials)
    
    #laser trials
    if  isfield(TE.Custom,'LaserTrial') && sum(TE.Custom.LaserTrial)>0
    if isfield (TE.Custom,'LaserTrialTrainStart')
    TEbis.LaserTrialTrainStart = TE.Custom.LaserTrialTrainStart(1:TEbis.nTrials)
    TEbis.LaserTrialTrainStartAbs = TEbis.LaserTrialTrainStart+TEbis.ResponseStart
    TEbis.LaserTrial =double( TE.Custom.LaserTrial(1:TEbis.nTrials))
    TEbis.LaserTrial (TEbis.CompletedTrial==0)=0
    TEbis.LaserTrial (TEbis.LaserTrialTrainStartAbs>TEbis.ResponseEnd)=0
    TEbis.LaserTrialTrainStartAbs(TEbis.LaserTrial~=1)=NaN
    TEbis.LaserTrialTrainStart (TEbis.LaserTrial~=1)=NaN
    
    TEbis.CompletedWTLaserTrial = TEbis.LaserTrial
    TEbis.CompletedWTLaserTrial(TEbis.CompletedWTTrial~=1)=NaN
    else #old version, laser during entire time investment
    TEbis.LaserTrialTrainStart=zeros(1,TEbis.nTrials)
    TEbis.LaserTrialTrainStartAbs=TEbis.ResponseStart
    TEbis.LaserTrial =double( TE.Custom.LaserTrial(1:TEbis.nTrials))
    TEbis.LaserTrial (TEbis.CompletedTrial==0)=0
    TEbis.LaserTrialTrainStartAbs(TEbis.LaserTrial~=1)=NaN
    TEbis.LaserTrialTrainStart (TEbis.LaserTrial~=1)=NaN
    end
    
    else
    TEbis.LaserTrialTrainStart = nan(1,TEbis.nTrials)
    TEbis.LaserTrialTrainStartAbs = nan(1,TEbis.nTrials)
    TEbis.LaserTrial = zeros(1,TEbis.nTrials)
    TEbis.CompletedWTLaserTrial = nan(1,TEbis.nTrials)
    TEbis.CompletedWTLaserTrial(TEbis.CompletedWTTrial==1) = 0
    end
    
    
    save(fullfile(Directory,'TEbis.mat'),'TEbis')
    save(fullfile(Directory,  'TrialEvents.mat'),'-struct','TEbis')
    
    disp('Additional events created and Trial Event saved')
    """