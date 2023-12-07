"""
Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import tqdm

import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
import os
import glob
import pickle
import scipy.io as sio
from datetime import date
from scipy.signal import find_peaks


def align_spikes_to_event(event_name, prebuffer, postbuffer, behav_df, traces, metadata, sps):
    """
    returns: array [int] of spiking activity [n_neurons x n_trials x pre_event + post_event]
    where pre_event and post_event are prebuffer and postbuffer (in seconds) converted to samples
    """
    behav_df = behav_df[~np.isnan(behav_df['TTLTrialStartTime'])]

    # trial start in bins (index) in the recording system timeframe
    t_starts_ms = np.round(sps * behav_df['TTLTrialStartTime']).astype('int')
    event_time_after_start = np.round(sps*behav_df[event_name]).astype('int')
    assert len(event_time_after_start) == len(t_starts_ms)
    event_time = t_starts_ms + event_time_after_start
    pre_event = int(sps*prebuffer)
    post_event = int(sps*postbuffer)

    n_trials = len(behav_df)
    n_neurons = traces.shape[0]
    spikes = np.zeros([n_neurons, n_trials, pre_event + post_event]).astype('uint8')

    for i in range(n_trials):
        spikes[:, i, :] = traces[:, (event_time[i]-pre_event):(event_time[i] + post_event)]

    return spikes


def trial_start_align(behav_df, traces, metadata, sps, max_allowable_len=36000):
    for red_flag in ['no_matching_TTL_start_time', 'large_TTL_gap_after_start']:
        if red_flag in behav_df.keys() and behav_df[red_flag].sum()>0:
            print('Trials with' + red_flag + '!!!')

    if np.isnan(behav_df['TTLTrialStartTime']).sum():
        print('Removing trials without TTL start times.')
        behav_df = behav_df[~np.isnan(behav_df['TTLTrialStartTime'])]

    # ----------------------------------------------------------------------- #
    # Find longest trial, so that all trials can be zero padded to same len
    # ----------------------------------------------------------------------- #
    # number of bins in a trial
    trial_len = np.ceil(sps*(behav_df['TrialLength']).to_numpy()).astype('int')

    longest_trial = max(trial_len)


    if longest_trial < 12000:
        longest_trial = 12000
    elif longest_trial < 24000:
        longest_trial = 24000
    elif longest_trial < max_allowable_len:
        longest_trial = max_allowable_len
    else:
        # In this case, the longest trial is longer than some arbitrary limit 
        # and will be truncated. However, the next-longest may also be longer 
        # than the upper limit, so all trials that are longer than the limit 
        # need to be truncated.
        # This is not the case above, where extending the longest_trial does 
        # not change that it is the longest.
        print("Warning: Truncating very long trials.")

        longest_trial = max_allowable_len
        trial_len[trial_len > max_allowable_len] = longest_trial
        
        long_idx = (sps * behav_df['TrialLength']) > max_allowable_len
        behav_df.loc[long_idx, 'TrialLength'] = longest_trial / sps  # store in s
    # ----------------------------------------------------------------------- #

    # trial start in bins (index) in the recording system timeframe
    t_starts = np.round(sps*behav_df['TTLTrialStartTime']).astype('int')

    n_trials = len(behav_df)
    n_neurons = traces.shape[0]
    spikes = np.zeros([n_trials, n_neurons, longest_trial]).astype('uint8')

    for i in range(n_trials):
        # print(i, t_starts[i],  trial_len[i], traces.shape)
        spikes[i, :, 0:trial_len[i]] = traces[:, t_starts[i]:(t_starts[i] + trial_len[i])]

    # rearrange spiking data from
    # [n_trials x n_neurons x time] -> [n_neurons x n_trials x time]
    spikes = spikes.transpose(1, 0, 2)

    return spikes, behav_df


def create_traces_np(behav_df, traces, metadata,
                     traces_aligned='ResponseStart',
                     aligned_ind=40,
                     filter_by_trial_num=False,
                     preITI=1.5):
    '''
    create different alignments from neuropixels data. The standard alignments are:
    stim_aligned
    resp_aligned
    reward_aligned
    interp_traces

    returns:
    dictionary with fields:
    stim_aligned
    resp_aligned
    reward_aligned
    interp_traces

    inds_interp

    time_ra
    time_stim
    time_rew
    time_interp

    '''
    sps = metadata['sps']
    n_neurons, n_trials, n_time_bins  = traces.shape

    # only sending in completed trials now
    if filter_by_trial_num:
        trial_number = behav_df['TrialNumber'].to_numpy() -1
    else:
        trial_number = behav_df.index.to_numpy()

    assert trial_number.size == n_trials
    print(f'Check: the trial number in the spike and behavioral data match: {trial_number.size}, {n_trials}')

    trial_len_arr = behav_df['TrialLength'].to_numpy()

    # -------------------------------------------------------------------- #
    #                     Trial events of interest                         #
    # -------------------------------------------------------------------- #
    center_poke = behav_df['PokeCenterStart'].to_numpy()  # trial start from animal's (not BPod's) perspective
    # between center poke and stimulus presentation is a prestim delay (drawn from uniform distr.)
    stim_on = behav_df['StimulusOnset'].to_numpy()
    stim_off = behav_df['StimulusOffset'].to_numpy()

    choice_port_entry = behav_df['ResponseStart'].to_numpy()
    resp_end = behav_df['ResponseEnd'].to_numpy()  # either leaves choice port or reward delivered


    if traces_aligned == "TrialStart":
        trial_len_in_bins = (trial_len_arr * sps).astype('int')

        center_poke_idx = np.round(center_poke * sps).astype('int')
        
        stim_on_idx = np.round(stim_on * sps).astype('int')
        stim_off_idx = np.round(stim_off * sps).astype('int')

        response_start_idx = np.round(choice_port_entry * sps).astype('int')
        resp_end_idx = np.round(resp_end * sps).astype('int')
    elif traces_aligned == 'ResponseStart':
        # next trial start is trial_len_arr - choice_port_entry
        trial_len_in_bins = (sps*(trial_len_arr - choice_port_entry) + aligned_ind).astype('int')

        # trial start (enters center poke) from animal's perspective but negative
        # because it's relative to the response start
        center_poke_idx = (aligned_ind - sps*(choice_port_entry - center_poke)).astype('int')
        
        # stimulus is response time before choice, which starts at aligned ind
        stim_on_idx = (aligned_ind - sps*(choice_port_entry - stim_on)).astype('int')
        stim_off_idx = (aligned_ind - sps*(choice_port_entry - stim_off)).astype('int')

        response_start_idx = np.array([int(aligned_ind)] * len(trial_number))

        # reward is waiting time after choice, which starts at aligned_ind
        # things are already response aligned
        resp_end_idx = (sps * (resp_end - choice_port_entry) + aligned_ind).astype('int')


    # ---------------------------------------------------------------------- #
    # Padding the traces with zeros such that all aligned traces
    # are the same length
    # ---------------------------------------------------------------------- #

    # Calculate padding size at beginning and end of trace
    earliest_start = np.min(center_poke_idx) - int(sps * preITI)

    if earliest_start < 0:
        prepad = int(np.ceil(abs(earliest_start)))
    else:
        prepad = 0

    longest_trial = np.max([trial_len_in_bins.max(), resp_end_idx.max() + int(2*sps)])

    if longest_trial > n_time_bins:
        postpad = int(np.ceil(longest_trial - n_time_bins))
    else:
        postpad = 0

    # Update the indices with the appropriate offset
    response_start_idx += prepad
    resp_end_idx += prepad
    stim_on_idx += prepad
    center_poke_idx += prepad
    stim_off_idx += prepad
    trial_len_in_bins += prepad

    # Add padding to the time axis of the traces [n_nrn, n_trials, time in bins]
    padded_traces = np.pad(traces[:, trial_number, :].astype('float'),
                           pad_width=[(0, 0), (0, 0), (prepad, postpad)],
                           mode='constant', constant_values=np.nan
                          )
    if not np.all(padded_traces == traces):
        print('Traces padded.')
    # --------------------------------------------------------------------- #


    # ---------------------------------------------------------------------- #
    # Calculate the aligned traces
    # ---------------------------------------------------------------------- #
    
    def align_helper(begin_arr, end_arr, index_arr, trace_len_bins):
        """
        Given the beginning and end of a certain type of event,
        this function finds the point at which that event is indexed
        and its delay from the beginning of the event.
        
        All traces for a given event are then aligned to a common reference
        point by their offset from the maximum delay.
        
        Parameters
        ----------
        begin_arr : [ARRAY] start times.
        end_arr : [ARRAY] end times. :P
        index_arr : [ARRAY] times at which event was registered.
        trace_len_bins : [int] common length of aligned traces

        Returns
        -------
        _aligned_arr : [ARRAY] aligned traces.
        reference_point : [INT] max delay from beginning, used as reference.

        """
        len_arr = end_arr - begin_arr
        
        delay_arr = index_arr - begin_arr  # bin number of event relative to beginning of frame
        reference_point = max(delay_arr)

        # number of bins at beginning of each trace such that the event lines up in all traces
        offset = (reference_point - delay_arr)

        _aligned_arr = np.full([n_trials, n_neurons, trace_len_bins], np.nan)
        for i in range(n_trials):
            _aligned_arr[i, :, offset[i]:(offset[i]+len_arr[i])] = padded_traces[:, i, begin_arr[i]:end_arr[i]]
            
        return _aligned_arr, reference_point
    
    
    # -----------------------Stimulus-aligned frame------------------------------- #
    stim_frame_begin = center_poke_idx - int(.5*sps)  # extend the beginning of the frame 0.5s before center poke
    # the end of the frame extends to 0.6s after the stimulus is on OR .15s after the stimulus ends
    stim_frame_end = stim_on_idx + int(0.5*sps) #np.minimum(stim_off_idx + int(.15*sps), stim_on_idx + int(.6*sps))

    stim_aligned, stim_point = align_helper(stim_frame_begin, stim_frame_end, stim_on_idx, trace_len_bins=int(2.5*sps))
    
    
    # -----------------------Response-aligned frame------------------------------- #
    # resp_frame_begin = np.maximum(stim_frame_end, response_start_idx - int(sps))
    resp_frame_begin = response_start_idx - int(2.1*sps)
    resp_frame_end = response_start_idx + int(3.1*sps)  # np.minimum(response_start_idx + int(10*sps), resp_end_idx + int(2.1*sps))

    response_aligned, response_point = align_helper(resp_frame_begin, resp_frame_end, response_start_idx, trace_len_bins=int(13.5*sps))


    # -----------------------Reward-aligned frame--------------------------------- #
    reward_frame_begin = np.maximum(response_start_idx, resp_end_idx - int(6.1*sps))
    reward_frame_end = resp_end_idx + int(2.1*sps)  # np.minimum(resp_end_idx + int(2.1*sps), trial_len_in_bins)
    
    reward_aligned, reward_point = align_helper(reward_frame_begin, reward_frame_end, resp_end_idx, trace_len_bins=int(10.1*sps))


    # -----------------------Interpolated----------------------------------- #
    # AKA time warping [see Williams et al., (2020), Neuron 105, 246–259 for a description of the problem]
    # will have the biggest effect on the response delay period
    interp_traces = []
    for i in range(n_trials):
        if (stim_on_idx[i] - center_poke_idx[i]) < 1:
            stim_on_idx[i] += 1
        if (stim_off_idx[i]) - stim_on_idx[i] <1:
            stim_off_idx[i] +=1

        # the interpolation frames are delineated by the different events and compose the entire trial
        interp_frames = [np.arange(center_poke_idx[i] - int(preITI*sps), center_poke_idx[i], dtype='int'),
                         np.arange(center_poke_idx[i], stim_on_idx[i], dtype='int'),
                         np.arange(stim_on_idx[i], stim_off_idx[i], dtype='int'),
                         np.arange(stim_off_idx[i], response_start_idx[i], dtype='int'),
                         np.arange(response_start_idx[i], response_start_idx[i] + .75*sps, dtype='int'),
                         np.arange(response_start_idx[i] + .75*sps, resp_end_idx[i], dtype='int'),
                         np.arange(resp_end_idx[i], trial_len_in_bins[i], dtype='int')]

        # print(poke_ind[i], stim_ind[i], move_ind[i], response_ind[i])
        # print(reward_ind[i], iti_ind[i])
        # print([len(f) for f in interp_frames])
        # print(padded_traces.shape)

        interp_lens = [int(0.5*sps),  # ITI
                       int(0.2*sps),  # stim_begin = 1
                       int(.35*sps),  # stim_end = 2
                       int(.15*sps),  # response_begin = 3
                       int(0.5*sps),  # arbitrary buffer?
                       int(3.0*sps),  # time investment (feedback delay), reward-bias will be stretched too
                       int(0.5*sps)]  # arbitrary buffer?

        # trace is matrix with [n_nrns x time in bins]
        trace_i = padded_traces[:, i, :].reshape([n_neurons, -1])
        interp_traces.append(create_trial_interp(trace_i, interp_frames=interp_frames, interp_lens=interp_lens))

    interp_traces = np.array(interp_traces)
    assert interp_traces.shape == (n_trials, n_neurons, sum(interp_lens))
    

    # ---------------------------------------------------------------------- #
    # Save traces and important variables used to create them
    # ---------------------------------------------------------------------- #
    traces_dict = {'stim_aligned': stim_aligned,
                   'stim_ind': stim_point,
                   'response_aligned': response_aligned,
                   'response_ind': response_point,
                   'reward_aligned': reward_aligned,
                   'reward_ind': reward_point,
                   'interp_traces': interp_traces,
                   'interp_inds': interp_lens}

    return traces_dict


def custom_interp(interp_axis, data_mat):
    """The data in each row of a data matrix is interpolated along a new axis that is common for all rows."""
    out = np.zeros((data_mat.shape[0], len(interp_axis)))
    data_idx = np.arange(data_mat.shape[1])
    for i, data_arr in enumerate(data_mat):
        if data_arr.sum():  # don't need to interpolate empty arrays
            out[i] = np.interp(interp_axis, data_idx, data_arr)
    return out


def create_trial_interp(traces, interp_frames, interp_lens):
    """The data within each frame is given a custom time axis."""
    ts = []
    for frame_idx, interp_period in zip(interp_frames, interp_lens):
        warped_frame = np.linspace(0, frame_idx.shape[0], interp_period)
        # take the traces over the original frame and project them onto the warped axis
        ts.append(custom_interp(warped_frame, traces[:, frame_idx]))
    ts_mat = np.concatenate(ts, axis=1)
    assert ts_mat.shape == (traces.shape[0], sum(interp_lens))
    return ts_mat


def get_trace_feature_df(behav_df, selected_neurons, traces, rat_name, session_date, probe_num,
                         behavior_variables=['prev_response_side1', 'prev_correct', 'stim_dir']):
    """
    Creates a dataframe of the spiking response ('activity') for all selected neurons and
    pairs the values of the given behavioral task variables with the spiking response in each time bin.

    :param behav_df: [pandas Dataframe] with values of behavioral variables from each trial
    :param selected_neurons: [numpy array] of neurons to use
    :param traces: [numpy array] spiking responses ('activity') with dimensions [#trials x #neurons x #time bins]
    :param behavior_variables: [list] names of task variables, such as trial outcome or wait time
    :param rat_name: [string]
    :return: [pandas Dataframe] each row contains the result of a single time bin within a given trial,
            or 'feature', with columns:
                - behavior variable values
                - neuron number
                - trial number
                - spiking response ('activity') of a single time bin
                - time bin number
                - rat name
    """
    dataframe_list = []
    for nrn in selected_neurons:
        all_traces = traces[:, nrn, :].astype('float32')  # trials, frame
        n_trials, n_frames = all_traces.shape

        # Concatenate all traces into one long array, with the time axis repeating: [trial 1, trial 2, ...]
        repeated_time_indices = np.tile(np.arange(n_frames), n_trials)
        traces_in_single_series = all_traces.flatten()

        # Concatenate all trial info into one long array, with the result for a single trial
        # repeated at each time step: [result_trial1 x n_frames, ..., result_trialN x n_frames]
        trials = np.repeat(behav_df.index.to_numpy(), n_frames)
        trial_results = [np.repeat(behav_df[code_name], n_frames) for code_name in behavior_variables]

        # Combine results
        dict_names = behavior_variables + ['activity', 'time', 'neuron', 'trial']
        vars_to_plot = trial_results + [traces_in_single_series, repeated_time_indices, nrn, trials]
        result_dict = dict(zip(dict_names, vars_to_plot))

        # Convert to dataframe and append to dataframe list
        dataframe_list.append(pd.DataFrame(result_dict))

    # Convert the dataframe list into single master dataframe
    df = pd.concat(dataframe_list).copy().reset_index()
    df['rat_name'] = rat_name
    df['session_date'] = session_date
    df['probe_num'] = probe_num

    return df






# LOAD DATA
def load_data(data_name: str, data_folder: str):
    """Load a data file from the rat priors project

    Args:
        data_name (string): filename of the specific folder corresponding to a dataset
        data_folder (string, optional): The folder where all the data lives. If None
                                        given then there is a default folder
                                        Z:\\Complete_data_files\\RDKPriorsTask\\

    Returns:
        dictionary: a dictionary of the data
    """

    if (data_folder is None):  # set this to some default path
        data_folder = 'Z:\\Complete_data_files\\RDKPriorsTask\\'

    pickle_file_path = os.path.join(data_folder, data_name) + '/*.pickle'
    data_file_name = glob.glob(pickle_file_path)
    print(pickle_file_path, data_file_name)
    assert len(data_file_name) == 1  # make sure this only points to one pickle file

    with open(data_file_name[0], 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    return data


# CREATE_INTERP_TRACES
def create_interp_traces(d, frame_rate=10, return_frame_amounts=False):
    # there will be multiple stages
    # 1) pre-nosepoke frames (no interp), 5 frames
    # 2) pre_stim frames (no interp), 5 frames
    # 3) stim frames (interp to 10 frames)
    # 4) stim end to resp (interp to 5 frames)
    # 5) post response (no iterp), 10 frames

    traces = np.array(d['all_data']['traces'])
    traces = traces.T if traces.shape[0] > traces.shape[1] else traces
    print('frame rate is ', frame_rate)

    was_completed = np.where(d['all_data']['was_completed'])[0]
    was_completed = np.array(was_completed, dtype=int)
    # print(type(was_completed),type(was_completed[0]))
    try:
        pre_stim_hold = int(d['all_data']['task_info']['pre_stim_hold'] * frame_rate)
    except:
        pre_stim_hold = .5 * frame_rate
        print('pre stim hold not found, assuming', pre_stim_hold)
    try:
        stim_hold = int(d['all_data']['task_info']['stim_hold'] * frame_rate)
    except:
        stim_hold = 1 * frame_rate
        print('stim hold not found, assuming', stim_hold)
    stim_start_frame = d['all_data']['frame_info']['stim_start_frame'][was_completed]
    response_frame = d['all_data']['frame_info']['response_frame'][was_completed]
    # nosepoke_frame = stim_start_frame - pre_stim_hold
    nosepoke_frame = d['all_data']['frame_info']['start_poke_frame'][was_completed]

    if np.sum(np.isnan(stim_start_frame)) > 0:
        print('found nans!! in stim_start_frame, ', np.sum(np.isnan(stim_start_frame)), 'trying to fix')
        print(stim_start_frame.shape)
        stim_start_frame[np.where(np.isnan(stim_start_frame))[0]] = nosepoke_frame[
            np.where(np.isnan(stim_start_frame))[0]]
        print(stim_start_frame.shape)

    if np.sum(np.isnan(nosepoke_frame)) > 0:
        print('found nans!! in nosepoke_frame, ', np.sum(np.isnan(stim_start_frame)), 'trying to fix')
        print(nosepoke_frame.shape)
        nosepoke_frame[np.where(np.isnan(nosepoke_frame))[0]] = stim_start_frame[
            np.where(np.isnan(stim_start_frame))[0]]
        print(stim_start_frame.shape)

    def custom_interp(x, yp):
        out = []
        for yp_ in [t_ for t_ in yp]:
            out.append(np.interp(x, np.arange(len(yp_)), yp_))
        return np.array(out)

    # print('pre-nosepoke is ', pre_stim_hold, ' frames. stimulus is ', stim_hold,'frames')

    interp_traces = []

    plt.figure()
    plt.subplot(121)
    plt.hist(nosepoke_frame[1:] - response_frame[0:-1], bins=100)
    plt.title(np.mean(nosepoke_frame[1:] - response_frame[0:-1]))
    plt.xlim([0, 100])  # only look at 10 seconds
    plt.subplot(122)
    plt.hist(response_frame - (stim_start_frame + 10), bins=100)
    plt.title(np.mean(response_frame - (stim_start_frame + 10)))
    plt.show()

    for t in range(len(stim_start_frame)):
        npf, ssf, rf = int(nosepoke_frame[t]), int(stim_start_frame[t]), int(response_frame[t])
        if t > 0:
            rf_last = int(response_frame[t - 1])
        else:
            rf_last = npf - 30  # on the first trial we just take the 3 seconds before the nosepoke
            rf_last = max(0, rf_last)

        if t + 1 < len(stim_start_frame):
            sf2 = int(nosepoke_frame[t + 1])
        else:
            sf2 = int(response_frame[t] + 30)  # on the last trial we take the 3 seconds after the response
            sf2 = min(traces.shape[1] - 1, sf2)  # traces is neurons x trials x time

        t1_ = traces[:, np.arange(rf_last, npf)]  # 30 pre-nosepoke frames
        t1_ = custom_interp(np.linspace(0, t1_.shape[1], 30), t1_)  # arbitrarily enforcing a 3 second ITI

        t2_ = traces[:, np.arange(npf, ssf)]  # pre-stim frames
        t2_ = custom_interp(np.linspace(0, t2_.shape[1], 5), t2_)  # prepoke frames, interp to 5

        t3_ = traces[:, np.arange(ssf, ssf + stim_hold)]  # stim frames, interp to 10
        t3_ = custom_interp(np.linspace(0, t3_.shape[1], 10), t3_)  # 35-45

        if len(np.arange(ssf + stim_hold, rf)) == 0:
            t4_ = np.nan * traces[:, np.arange(ssf + stim_hold, rf + 3)]  # dont want response info bleeding over!
        else:
            t4_ = traces[:, np.arange(ssf + stim_hold, rf)]  # movement to response, interp to 5

        t4_ = custom_interp(np.linspace(0, t4_.shape[1], 3), t4_)  # (40-45) #more accurate

        t5_ = traces[:, np.arange(rf, sf2)]  # next poke frame

        t5_ = custom_interp(np.linspace(0, t5_.shape[1], 30), t5_)  # 3 second ITI on the back side

        interp_traces.append(np.concatenate((t1_, t2_, t3_, t4_, t5_), axis=1))

        if return_frame_amounts == True:
            interp_frame_dict = {'pre_nosepoke': t1_.shape[1],
                                 'pre_stim_hold': t2_.shape[1],
                                 'stim_hold': t3_.shape[1],
                                 'move_frames': t4_.shape[1],
                                 'post_response': t5_.shape[1]}
    if return_frame_amounts == False:
        mm = stats.mode(np.array([i.shape for i in interp_traces])[:, 1]).mode[0]
        mm = np.min([i.shape[1] for i in interp_traces])
        interp_traces = [i[:, 0:mm] for i in interp_traces]
        plt.plot([i.shape[1] for i in interp_traces])
        return np.array(interp_traces)

    else:
        print('Pre stim frames is', pre_stim_hold, '\nStim hold frames is', stim_hold)
        stim_aligned_frame_dict = {'stim_start': 40,
                                   'stim_end': 40 + stim_hold,
                                   'pre_stim_start': 40 - pre_stim_hold}
        resp_aligned_frame_dict = {'resp_frame': 60}
        return interp_frame_dict, stim_aligned_frame_dict, resp_aligned_frame_dict


# ORGANIZE DATA
def organize_data(data, binarize=False, frame_rate=10):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    frame_rate : int, optional
        [description], by default 10

    Returns
    -------
    [type]
        [description]
    """

    data = [data] if type(data) == dict else data

    print('The number of datasets is', len(data))

    # create interpolated traces
    interp_traces = [create_interp_traces(d, frame_rate) for d in data]
    interp_traces = np.concatenate(interp_traces, axis=0)

    interp_frame_dict, stim_aligned_frame_dict, resp_aligned_frame_dict = create_interp_traces(data[0], frame_rate,
                                                                                               return_frame_amounts=True)
    print(interp_frame_dict, stim_aligned_frame_dict)
    # create stim aligned traces
    traces = [d['completed_trials_data']['traces_stim_aligned'] for d in data]

    traces_stim_aligned = np.concatenate(traces, axis=0)

    # create response aligned data
    traces = [d['completed_trials_data']['traces_resp_aligned'] for d in data]
    traces_resp_aligned = np.concatenate(traces, axis=0)

    # create traces from all data
    traces_all = [d['all_data']['traces'] for d in data]
    try:
        traces_all = np.concatenate(traces_all, axis=1)
    except:
        traces_all = np.concatenate(traces_all, axis=0)

    if binarize:
        print("im binarizing the traces")
        trials, neurons, _ = traces_stim_aligned.shape
        stim_spikes = np.zeros_like(traces_stim_aligned)

        resp_spikes = np.zeros_like(traces_resp_aligned)
        for t in range(trials):
            for n in range(neurons):
                # print(traces_stim_aligned[t, n, :].shape)
                a = find_peaks(traces_stim_aligned[t, n, :], prominence=1 * np.std(traces_stim_aligned[t, n, :]))[0]
                stim_spikes[t, n, a] = 1

                a = find_peaks(traces_resp_aligned[t, n, :], prominence=1 * np.std(traces_resp_aligned[t, n, :]))[0]
                resp_spikes[t, n, a] = 1

        traces_stim_aligned = stim_spikes
        traces_resp_aligned = resp_spikes

    data_ = []

    print('traces_all shape is ', traces_all.shape, 'traces_stim_aligned shape is ', traces_stim_aligned.shape)

    for i, d in enumerate(data):
        d_ = d['all_data']
        print(d_['frame_info']['stim_start_frame'].shape)
        print(d_.keys())
        df = pd.DataFrame()
        df['stim_start_frame'] = d_['frame_info']['stim_start_frame']
        df['stim_end_frame'] = d_['frame_info']['stim_end_frame']
        df['prev_stim_end'] = df['stim_end_frame'].shift(1)
        df['frame_from_prev'] = df['stim_start_frame'] - df['prev_stim_end']

        d_['stim_start_frame'] = df['stim_start_frame'].to_numpy()
        d_['stim_end_frame'] = df['stim_end_frame'].to_numpy()
        d_['prev_stim_end'] = df['prev_stim_end'].to_numpy()
        d_['frame_from_prev'] = df['frame_from_prev'].to_numpy()

        del d_['frame_info']
        del d_['task_info']
        del d_['traces']
        try:
            del d_['patterns']
            del d_['pattern_inds']
        except:
            pass

        num_trials = len(d_['was_completed'])

        for key, value in enumerate(d_):
            print(value, d_[value].shape)
            d_[value] = d_[value][0:num_trials]
            print(value, d_[value].shape)

        d_ = pd.DataFrame.from_dict(d_)
        d_['session'] = i
        print('hello world')
        data_.append(d_)

    data_ = pd.concat(data_)

    print(data_.keys())
    print(data_)
    data_ = data_.replace([3], [-1])
    try:
        data_ = data_.replace(['right'], [1])
    except:
        print('ERROR WHEN REPLACING STRING RIGHT')
    data_ = data_.replace([1], [1])
    try:
        data_ = data_.replace(['left'], [-1])
    except:
        print('ERROR WHEN REPLACING STRING LEFT')

    for k in data_:
        print(k, data_[k].unique())
    # check if a column is [1,3] or [90,270]

    # data_['prev_response_side']=data_.response_side.shift(1)
    # data_['prev_response_side2']=data_.response_side.shift(2)
    # data_['prev_stim_side']=data_.stim_dir.shift(1)

    # prev_response_side = data_.prev_response_side.tolist()
    # prev_stim_side = data_.prev_stim_side.tolist()

    # for i in range(len(prev_response_side)):
    #    j = prev_response_side[i]
    #    if j != 'left' or j != 'right':
    #        prev_response_side[i] = prev_stim_side[i]

    rs_ = data_.response_side.tolist()
    ss_ = data_.stim_dir.tolist()
    time_ = np.ones_like(ss_)
    data_['time'] = time_
    print(np.unique(rs_))
    n_replaced = 0
    for i in range(len(rs_)):
        j = rs_[i]
        if j == '0' or j == 0:
            n_replaced += 1
            rs_[i] = ss_[i]
    print('the number of response sides replaced for use in previous response sides is', n_replaced)
    rs_ = np.array(rs_)
    rs_[rs_ == 'left'] = -1
    rs_[rs_ == 'right'] = 1
    rs_ = [float(x) for x in rs_]

    # make variables for prev_avg_response_side_n
    for i in np.arange(1, 50):
        prev_sum_i = [np.mean(rs_[(trial - i):trial]) for trial in np.arange(0, len(rs_))]
        for ind, x in enumerate(prev_sum_i):
            if np.isnan(x):
                prev_sum_i[ind] = np.mean(rs_[0:ind])
        prev_sum_i[0] = 0.0
        data_['prev_avg_response_side' + str(i)] = prev_sum_i
        data_['prev_avg_response_side' + str(i)] = [float(x) for x in data_['prev_avg_response_side' + str(i)].tolist()]

    for i in np.arange(1, 50):
        data_['prev_response_side' + str(i)] = np.roll(rs_, i)
        data_['prev_response_side' + str(i)] = [float(x) for x in data_['prev_response_side' + str(i)].tolist()]

    ss_ = np.array(ss_)
    ss_[ss_ == 'left'] = -1
    ss_[ss_ == 'right'] = 1
    ss_ = np.array([float(x) for x in ss_])
    for i in np.arange(1, 50):
        data_['prev_stim_side' + str(i)] = np.roll(ss_, i)

    # make variables for prev_avg_response_side_n
    for i in np.arange(1, 50):
        prev_sum_i = [np.mean(ss_[(trial - i):trial]) for trial in np.arange(0, len(ss_))]
        for ind, x in enumerate(prev_sum_i):
            if np.isnan(x):
                prev_sum_i[ind] = np.mean(ss_[0:ind])
        prev_sum_i[0] = 0.0
        data_['prev_avg_stim_side' + str(i)] = prev_sum_i
        data_['prev_avg_stim_side' + str(i)] = [float(x) for x in data_['prev_avg_stim_side' + str(i)].tolist()]

    data_['signed_noise'] = data_['noise'].to_numpy() * ss_

    for cum_n in np.arange(1, 50):
        _ = np.array([np.roll(rs_, i + 1) for i in range(cum_n)])
        data_['resp_cum' + str(cum_n)] = np.sum(_.astype(int), axis=0)

    # data_['prev_response_side'] = prev_response_side

    # here we are going to compute if the previous trial was completed_trials_data
    data_['prev_was_completed'] = data_['was_completed'].shift(1)

    data = data_[data_['was_completed'] == 1]

    print('The number of trials, neurons, and frames is ', traces_stim_aligned.shape)
    print('The number of trials, neurons, and interp_frames is ', interp_traces.shape)

    print('The number of trials ', len(data))

    n_trials, n_neurons, n_frames = traces_stim_aligned.shape
    print(data)

    try:
        data = data.replace(['left', 'right'], [-1, 1])
    except:
        print('ERROR WHEN REPLACING LEFT RIGHT')

    resp_side = data.response_side.to_list()

    try:
        data['prior_side'] = 1 * (data['prior'] > 0.5)
    except:
        pass

    data['resp_rightward'] = data['response_side'].replace({-1: 0})

    # add delete rows with nan
    print(data.keys())

    return (data, traces_stim_aligned, traces_resp_aligned, traces_all, interp_traces,
            interp_frame_dict, stim_aligned_frame_dict, resp_aligned_frame_dict)


def bin_coherence_data(data):
    # Bin the continous coherence values
    # TODO: make this a function, and have binned_signed_noise and binned_noise as extra
    # TODO: variables
    data['raw_signed_noise'] = data['signed_noise']
    data.loc[:, 'signed_noise'] = pd.cut(data['signed_noise'],
                                         [-1.01, -1, -0.81, -0.5, -0.3, -0.1, 0.1, .3, .5, .81, 1.1])
    data.loc[:, 'signed_noise'] = data.loc[:, 'signed_noise'].apply(lambda x: x.mid.astype(float))
    data['signed_noise'] = data['signed_noise'].astype(float)

    data['raw_noise'] = data['noise']
    data.loc[:, 'noise'] = pd.cut(data['noise'], [-0.1, 0.1, .3, .5, .81, 1.1])
    data.loc[:, 'noise'] = data.loc[:, 'noise'].apply(lambda x: x.mid.astype(float))
    data['noise'] = data['noise'].astype(float)
    return data


def nonzero_the_traces(traces_stim_aligned, traces_resp_aligned, traces_all, interp_traces):
    for x in [traces_stim_aligned, traces_resp_aligned, traces_all, interp_traces]:
        x[x < 0] = 0
    return traces_stim_aligned, traces_resp_aligned, traces_all, interp_traces


def load_cell_COMs(data_name: str, data_folder: str):
    # Load filters
    directory = os.path.join(data_folder, data_name)
    depth_data = np.squeeze(sio.loadmat(directory + '/depth_data.mat')['depth_data'])  # in mm
    depth_data = depth_data - np.min(depth_data)

    com_data = np.squeeze(sio.loadmat(directory + '/com_data.mat')['com_data'])
    com_data.shape  # neurons,2

    return depth_data, com_data


# Put everything into one function
def import_and_organize_data(data_name: str, data_folder: str):


    data = load_data(data_name, data_folder)

    data, traces_stim_aligned, traces_resp_aligned, \
    traces_all, interp_traces, interp_frame_dict, \
    stim_aligned_frame_dict, resp_aligned_frame_dict = organize_data(data, frame_rate=10, binarize=False)

    data = bin_coherence_data(data)

    traces_stim_aligned, traces_resp_aligned, traces_all, \
    interp_traces = nonzero_the_traces(traces_stim_aligned,
                                       traces_resp_aligned,
                                       traces_all, interp_traces)

    data_dict = {'data': data, 'traces_sa': traces_stim_aligned, 'traces_ra': traces_resp_aligned,
                 'traces': traces_all, 'interp_frames': interp_frame_dict,
                 'traces_int': interp_traces, 'sa_frames': stim_aligned_frame_dict,
                 'ra_frames': resp_aligned_frame_dict}


    return data_dict  # (data, traces_stim_aligned, traces_resp_aligned, traces_all,\
    # interp_traces, interp_frame_dict, stim_aligned_frame_dict, resp_aligned_frame_dict)


def make_dir_if_not_exist(path):
    """create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)



def split_multiday(behav_df, traces_dict):
    sessions = sorted(behav_df.session.unique())

    dfs = []
    traces_dicts = []
    priors = []
    dates = []

    for session in sessions:
        sess_mask = (behav_df.session == session)
        #spoofing the date for now
        dates.append('None')
        df = behav_df[sess_mask]
        dfs.append(df.reset_index())

        traces_dicts.append({'interp_traces': traces_dict['interp_traces'][sess_mask],
                             'stim_aligned':traces_dict['stim_aligned'][sess_mask],
                             'response_aligned': traces_dict['response_aligned'][sess_mask],
                             'reward_aligned': traces_dict['reward_aligned'][sess_mask],
                             #python 3.7+ dictionary order is gauranteed
                             'interp_inds': traces_dict['interp_frames'],
                             'reward_ind': traces_dict['reward_frames'],
                             'stim_ind': traces_dict['stim_frames'],
                             'response_ind': traces_dict['response_frames']})

        #if there is more than one prior, there is a prior
        priors.append(len(df.prior.unique()) > 1)

    return priors, dates, sessions, dfs, traces_dicts



def filter_stable(obj, set_stable=False, n_bootstraps=10, cor_val=.95, save=False):

    _, n_neurons, trial_len = obj.interp_traces.shape

    trials = obj.behav_df[~obj.behav_df.rewarded].index.to_numpy()

    n_trials = len(trials)

    n_test = np.floor(n_trials / 2).astype('int')

    all_cors = []

    for i in range(n_bootstraps):
        trials = np.random.permutation(trials)
        # todo should be able to filter data_obj with an array
        test = list(trials[0:n_test].astype('int'))
        train = list(trials[n_test:].astype('int'))

        traces_test = obj[test, :, :].mean(axis=0)
        traces_train = obj[train, :, :].mean(axis=0)

        corr = [spearmanr(traces_test[n], traces_train[n]) for n in range(n_neurons)]
        corr = [r for (r, p) in corr]
        all_cors.append(corr)

    all_cors = np.array(all_cors).mean(axis=0)

    stable = all_cors > cor_val

    if set_stable:
        update_stable(obj, stable, save)

    return obj


def update_stable(obj, stable, save=True):
    obj.stable_neurons = np.where(stable)[0]
    print("Updated stable inds for rat: " + obj.metadata['rat_name'], end='')

    if save:
        obj.to_pickle()
        print(", and saved obj.")
    else:
        print(".")


def compute_d_primes(traces, var_data, n_shuff = 1000, distance = "dprime", balance_by = None):

    var_data2 = balance_by

    # first compute normal d-prime
    unique_vals = sorted(np.unique(var_data)) #make sure that direction of coding is always consistent
    assert len(unique_vals)==2, print('!there should only be 2 unique variable values!',unique_vals)
    # for each unique value we want to balance by another variable
    rus = RandomUnderSampler()
    ind1 = np.where(var_data==unique_vals[0])[0]
    ind2 = np.where(var_data==unique_vals[1])[0]

    # under sample 100 times
    d_prime = []
    coding_dir = []
    for i in tqdm.tqdm(range(100)):
        if balance_by is not None:
            X,y = rus.fit_resample(traces[ind1,0,:], var_data2[ind1])
            ind1_us = ind1[rus.sample_indices_]
            X,y = rus.fit_resample(traces[ind2,0,:], var_data2[ind2])
            ind2_us = ind2[rus.sample_indices_]
        else:
            ind1_us = ind1#np.random.choice(ind1, replace= True, size = int(len(ind1)*.9))
            ind2_us = ind2#np.random.choice(ind2, replace= True, size = int(len(ind2)*.9))

        d1 = traces[ind1_us,:,:]
        d2 = traces[ind2_us,:,:]
        if distance == "dprime" or (traces.shape[1] == 1):
            d_prime.append(compute_d_prime(d1,d2))
        elif distance == "mahalanobis":
            try:
                d_prime.append(compute_mahalanobis(d1, d2))
            except:
                d_prime.append(np.nanmean(compute_d_prime(d1, d2), axis = 1))

        coding_dir.append((d1.mean(axis = 0) - d2.mean(axis = 0))*2/(d1.mean(axis = 0) + d2.mean(axis = 0)))

    # now shuffle label and compute
    n_trials = len(var_data)
    shuf_inds = [np.random.permutation(var_data) for i in range(n_shuff)] #1000, n_trials

    # for every row of shuf_inds, we want to get all the interp_traces of one type

    d_primes_shuff = []
    rus = RandomUnderSampler()

    for i in tqdm.tqdm(shuf_inds):

        ind1 = np.where(i==unique_vals[0])[0]
        ind2 = np.where(i==unique_vals[1])[0]
        if balance_by is not None:
            X,y = rus.fit_resample(traces[ind1,0,:], var_data2[ind1])
            ind1_us = ind1[rus.sample_indices_]
            X,y = rus.fit_resample(traces[ind2,0,:], var_data2[ind2])
            ind2_us = ind2[rus.sample_indices_]
        else:
            ind1_us = ind1#np.random.choice(ind1, replace= True, size = int(len(ind1)/2))
            ind2_us = ind2#np.random.choice(ind2, replace= True, size = int(len(ind1)/2))


        d1 = traces[ind1_us,:,:]
        d2 = traces[ind2_us,:,:]

        if (distance == "dprime") or (traces.shape[1] == 1):
            d_primes_shuff.append( compute_d_prime(d1,d2) )# neurons,frames
        elif distance == "mahalanobis":
            try:
                d_primes_shuff.append(compute_mahalanobis(d1, d2))
            except:
                d_primes_shuff.append(compute_d_prime(d1, d2))  # neurons,frames


    p_vals = np.array([np.mean(i < np.array(d_primes_shuff),axis=0) for i in d_prime])
    p_vals_max = np.max(p_vals,axis=0)
    p_vals_mean = np.mean(p_vals,axis=0)

    return {'d_prime':d_prime, 'd_prime_shuff':d_primes_shuff, 'p_vals':p_vals,
                'p_vals_max':p_vals_max, 'p_vals_mean':p_vals_mean, 'coding_dir': coding_dir}

def compute_d_prime(d1,d2):
    means = [np.nanmean(d1, axis=0),np.nanmean(d2, axis=0)]
    stds = [np.nanstd(d1, axis=0), np.nanstd(d2, axis=0)]
    d_prime = np.abs(means[0] - means[1])/np.sqrt(0.5*(stds[0]**2+stds[1]**2)) # neurons,frames
    return d_prime


from scipy.spatial import distance
def compute_mahalanobis(d1, d2):
    cat = np.r_[d1.reshape(d1.shape[0:2]), d2.reshape(d2.shape[0:2])]
    V = np.cov(cat, rowvar = False)

    IV = np.linalg.pinv(V)
    return distance.mahalanobis(d1.mean(axis = 0), d2.mean(axis = 0), IV)
