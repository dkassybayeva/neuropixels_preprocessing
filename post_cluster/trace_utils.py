"""
Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import tqdm

def trial_start_align(behav_df, traces, sps):

    behav_df = behav_df[~np.isnan(behav_df.TrialStartAligned)]

    # trial start in bins (index)
    t_starts = np.round(sps*behav_df['TrialStartAligned']).astype('int')
    
    # number of bins in a trial
    trial_len = np.ceil(sps*(behav_df['trial_len']).to_numpy()).astype('int')

    longest_trial = max(trial_len)

    if longest_trial < 12000:
        longest_trial = 12000
    elif longest_trial < 24000:
        longest_trial = 12000
    elif longest_trial < 36000:
        longest_trial = 36000
    else:
        longest_trial = 36000

        longest_ind  = (behav_df['trial_len']).argmax().to_numpy()
        trial_len[longest_ind] = longest_trial

        behav_df.loc[(behav_df['trial_len']).argmax(), 'trial_len'] = longest_trial
        print(behav_df.loc[(behav_df['trial_len']).argmax(), 'trial_len'])
        print("warning, truncating a very long trial")

    n_trials = len(behav_df)
    n_neurons = traces.shape[0]

    spikes = np.zeros([n_trials, n_neurons, longest_trial]).astype('uint8')

    for i in range(n_trials):
        print(i, t_starts[i],  trial_len[i], traces.shape)
        spikes[i, :, 0:trial_len[i]] = traces[:, t_starts[i]:(t_starts[i] + trial_len[i])]

    # rearrange spiking data from
    # [n_trials x n_neurons x time] -> [n_neurons x n_trials x time]
    spikes = spikes.transpose(1, 0, 2)

    return spikes, behav_df


def create_traces_np(behav_df, traces, sps,
                     traces_aligned='ResponseStart',
                     aligned_ind=40,
                     filter_by_trial_num=False,
                     preITI=.5): 
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
    n_neurons, n_trials, n_time_bins  = traces.shape

    #behav_df.loc[behav_df['TrialNumber'] == max(behav_df['TrialNumber']), 'next_trial_start'] = behav_df.loc[behav_df['TrialNumber'] == max(behav_df['TrialNumber']), 'TrialStartAligned']  + 60


    #only sending in completed trials now
    if filter_by_trial_num:
        trial_number = behav_df['TrialNumber'].to_numpy() -1
    else:
        trial_number = behav_df.index.to_numpy()

    assert trial_number.size == n_trials
    print(max(trial_number) + 1, n_trials)


    resp_on = behav_df['ResponseStart'].to_numpy()
    resp_off = behav_df['ResponseEnd'].to_numpy()
    center_poke = behav_df['PokeCenterStart'].to_numpy()
    
    stim_on = behav_df['StimulusOnset'].to_numpy()
    stim_off = behav_df['StimulusOffset'].to_numpy()
    trial_len_arr = behav_df['trial_len'].to_numpy()

    if traces_aligned == "TrialStart":
        response_ind = np.round(resp_on * sps).astype('int')
        reward_ind = np.round(resp_off * sps).astype('int')
        poke_ind = np.round(center_poke * sps).astype('int')
        
        stim_ind = np.round(stim_on * sps).astype('int')
        move_ind = np.round(stim_off * sps).astype('int')
        iti_ind = (trial_len_arr * sps).astype('int')

    if traces_aligned == 'ResponseStart':
        response_ind = np.array([int(aligned_ind)] * len(trial_number))
        
        # reward is waiting time after choice, which starts at aligned_ind
        # things are already response aligned
        reward_ind = (sps * (resp_off - resp_on) + aligned_ind).astype('int') 
        poke_ind = (aligned_ind - sps*(resp_on - center_poke)).astype('int')
        
        # stimulus is response time before choice, which starts at aligned ind
        stim_ind = (aligned_ind - sps*(resp_on - stim_on)).astype('int')
        move_ind = (aligned_ind - sps*(resp_on - stim_off)).astype('int')
        
        # next trial start is trial_len_arr - resp_on
        iti_ind = (sps*(trial_len_arr - resp_on) + aligned_ind).astype('int')


    # ---------------------------------------------------------------------- #
    # Padding the traces with zeros such that all aligned traces 
    # of the same type are the same length
    # ---------------------------------------------------------------------- #
    
    # Calculate padding size at beginning and end of trace
    smin = np.min(poke_ind) - int(sps * preITI)

    if smin < 0:
        prepad = int(np.ceil(abs(smin)))
    else:
        prepad = 0

    imax = np.max([int(np.max(iti_ind)), np.max(reward_ind) + int(2*sps)])

    if imax > n_time_bins:
        postpad = int(np.ceil(imax - n_time_bins))
    else:
        postpad = 0

    # Update the indices with the appropriate offset
    response_ind += prepad
    reward_ind += prepad
    stim_ind += prepad
    poke_ind += prepad
    move_ind += prepad
    iti_ind += prepad

    # Add padding to the time axis of the traces
    padded_traces = np.pad(traces[:, trial_number, :], 
                           pad_width=[(0, 0), (0, 0), (prepad, postpad)], 
                           mode='empty'
                          ).astype('uint8')
    # --------------------------------------------------------------------- #


    # ---------------------------------------------------------------------- #
    # Calculate the aligned traces
    # ---------------------------------------------------------------------- #
    
    def align_helper(begin_arr, end_arr, index_arr, n_bins):
        len_arr = end_arr - begin_arr
        register_len = index_arr - begin_arr
        register_point = max(register_len)
        s = (register_point - register_len)
        
        _aligned_arr = np.zeros([n_trials, n_neurons, n_bins], dtype='uint16')
        for i in range(len(begin_arr)):
            _temp = padded_traces[:, i, begin_arr[i]:end_arr[i]]
            _aligned_arr[i, :, s[i]:(s[i]+len_arr[i])] = _temp
            
        return _aligned_arr
    
    # -----------------------Stimulus aligned------------------------------- #
    stim_begin = np.array([poke_ind[i] - int(.5*sps) for i in range(len(stim_ind))])
    stim_end = np.array([min(move_ind[i] + int(.15*sps), stim_ind[i] + int(.6*sps)) for i in range(len(stim_ind))])
    stim_len = stim_end - stim_begin
    poke_len = stim_ind - stim_begin
    stim_point = max(poke_len)
    s = (stim_point - poke_len)
    
    stim_aligned = (np.zeros([ len(trial_number), n_neurons, int(2.5*sps)])*np.nan).astype('uint16')
    for i in range(len(stim_begin)):
        stim_aligned[i, :, s[i]:(s[i] + stim_len[i])] =  padded_traces[:, i, stim_begin[i] :stim_end[i] ]
    
    # ------------
    stims = range(len(stim_ind))
    stim_begin2 = np.array([poke_ind[i] - int(.5*sps) for i in stims])
    
    move_plus = move_ind + int(.15*sps)
    stim_ind_plus =  stim_ind + int(.6*sps)
    stim_end2 = np.array([min(move_plus[i], stim_ind_plus[i]) for i in stims])
    
    stim_aligned2 = align_helper(stim_begin2, stim_end2, stim_ind, int(2.5*sps))
    assert np.all(stim_aligned == stim_aligned2)
    
    
    
    
    
    # -----------------------Response aligned------------------------------- #
    response_begin = np.array([max(stim_end[i], response_ind[i] - int(sps)) for i in range(len(stim_end))])
    response_end = np.array([min(response_ind[i]+ int(10*sps), reward_ind[i] +int(2*sps)) for i in range(len(response_ind))])

    response_len = response_end - response_begin
    pre_resp_len = response_ind - response_begin
    choice_point = max(pre_resp_len)
    r = (choice_point - pre_resp_len)
    
    response_aligned = (np.zeros([len(trial_number), n_neurons,  int(13.5*sps)])*np.nan).astype('uint16')
    for i in range(len(stim_begin)):
        response_aligned[i, :, r[i]: r[i] + response_len[i]] = padded_traces[:, i, response_begin[i]:response_end[i]]


    # ------------
    response_aligned2 = align_helper(response_begin, response_end, 
                                     response_ind, int(13.5*sps))
    assert np.all(response_aligned == response_aligned2)

    # -----------------------Reward aligned--------------------------------- #
    reward_begin = np.array([max(response_ind[i], reward_ind[i] - int(8*sps)) for i in range(len(response_end))])
    # print([iti_ind[i] for i in range(len(iti_ind))])
    # print([reward_ind[i] for i in range(len(iti_ind))])
    reward_end = np.array([min(reward_ind[i] + int(2*sps), iti_ind[i]) for i in range(len(iti_ind))])

    reward_len = reward_end - reward_begin
    WT_len = reward_ind - reward_begin
    reward_point = max(WT_len)
    w = reward_point - WT_len

    reward_aligned = (np.zeros([ len(trial_number), n_neurons, int(10.1*sps)])*np.nan).astype('uint16')
    for i in range(len(stim_begin)):
        reward_aligned[i, :, w[i]: w[i] + reward_len[i]] = padded_traces[:, i, reward_begin[i]:reward_end[i]]

    # ----------------
    reward_aligned2 = align_helper(reward_begin, reward_end, 
                                   reward_ind, int(10.1*sps))
    
    assert np.all(reward_aligned == reward_aligned2)
    

    # -----------------------Interpolated aligned--------------------------- #
    interp_traces = []
    for i in range(len(stim_ind)):
        if (stim_ind[i] - poke_ind[i]) < 1:
            stim_ind[i] += 1
        if (move_ind[i]) - stim_ind[i] <1:
            move_ind[i] +=1

        interp_frames = [np.arange(poke_ind[i] - int(preITI*sps), poke_ind[i]).astype('int'),
                         np.arange(poke_ind[i], stim_ind[i]).astype('int'),
                         np.arange(stim_ind[i], move_ind[i]).astype('int'),
                         np.arange(move_ind[i], response_ind[i]).astype('int'),
                         np.arange(response_ind[i], response_ind[i] + .75*sps).astype('int'),
                         np.arange(response_ind[i] + .75*sps, reward_ind[i]).astype('int'),
                         np.arange(reward_ind[i], iti_ind[i]).astype('int')]

        print(poke_ind[i], stim_ind[i], move_ind[i], response_ind[i])
        print(reward_ind[i], iti_ind[i])
        print([len(f) for f in interp_frames])
        print(padded_traces.shape)

        interp_lens = [int(.5*sps),
                       int(.2*sps),#stim_begin = 1
                       int(.35*sps),#stim_end = 2
                       int(.15*sps),#response_begin = 3
                       int(.5*sps),
                       int(3*sps),
                       int(.5*sps)]

        interp_traces.append(create_trial_interp(padded_traces[:, i, :].reshape([n_neurons, -1]), interp_frames = interp_frames, interp_lens = interp_lens))

    interp_traces = np.array(interp_traces)
    

    # ---------------------------------------------------------------------- #
    # Save traces and important variables used to create them
    # ---------------------------------------------------------------------- #
    traces_dict = {'interp_traces': interp_traces,
                   'stim_aligned': stim_aligned,
                   'response_aligned': response_aligned,
                   'reward_aligned': reward_aligned,
                   'stim_ind': stim_point,
                   'response_ind': choice_point,
                   'reward_ind': reward_point,
                   'interp_inds': interp_lens}

    return traces_dict


def custom_interp(x,yp):
    out = []
    for yp_ in [t_ for t_ in yp]:
        out.append(np.interp(x,np.arange(len(yp_)),yp_))
    return np.array(out).T

def create_trial_interp(traces, interp_frames, interp_lens):
    ts = [custom_interp(np.linspace(0,int_f.shape[0],il), traces[:, int_f]) for (int_f, il) in zip(interp_frames, interp_lens)]

    return np.concatenate(ts).T

def get_feature_df(behav_df, all_coding_inds, traces, code_names=['prev_response_side1', 'prev_correct', 'stim_dir'], rat_name = 'none'):
    d_all = []
    for nrn in all_coding_inds:
        all_traces = traces[:, nrn, :].astype('float32')  # trials, frame

        n_trials, n_frames = all_traces.shape

        vars_to_plot = [np.repeat(behav_df[code_name], n_frames) for code_name in code_names]

        #todo use index of behav_df
        trials = np.repeat(range(n_trials), n_frames)
        #trials = np.repeat(behav_df.index.to_numpy(), n_frames)
        dff = np.reshape(all_traces, -1)  # trials,frames

        frames = np.tile(np.arange(n_frames), n_trials)

        dict_names = code_names + ['activity', 'time', 'neuron', 'trial']
        vars_to_plot = vars_to_plot + [dff, frames, nrn, trials]

        dict = {cn: vtp for (cn, vtp) in zip(dict_names, vars_to_plot)}

        d_ = pd.DataFrame(dict)

        d_all.append(d_)
    df = pd.concat(d_all).copy().reset_index()

    df['rat_name'] = rat_name

    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import pickle
import scipy.io as sio
from datetime import date
from scipy.signal import find_peaks


# pims


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


from scipy.stats import spearmanr

def filter_active(obj, set_active = False, n_bootstraps = 10, cor_val = .95, save = False):

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

    active = all_cors > cor_val

    if set_active:
        update_active(obj, active, save)

    return obj


def update_active(obj, active, save = True):
    act_inds = np.where(active)[0]
    obj.active_neurons = act_inds
    print("updated active inds for rat: " +  obj.name)

    if save:
        obj.to_pickle()
        print("saved new active inds")
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
