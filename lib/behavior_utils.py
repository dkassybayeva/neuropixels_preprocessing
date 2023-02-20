import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_df(data_dir):
    raw_dict = loadmat(data_dir)
    behav_dict = {key: np.array(raw_dict[key]).squeeze() for key in raw_dict if '__' not in key and key!='Settings'}
    behav_df = pd.DataFrame.from_dict(behav_dict)

    # so that we can use this code for session data that doesnt have catch trials!
    if 'CatchTrial' in behav_df.keys():
        behav_df.CatchTrial = behav_df.CatchTrial.astype('bool')
    if 'CompletedTrial' in behav_df.keys():
        behav_df.CompletedTrial = behav_df.CompletedTrial.astype('bool')

    return behav_df

def trim_df(behav_df):
    '''
    its obnoxious to have a bunch of nan fields, would like to dropna, so just returning only the fields I care about and
    use. Presimably this will make pandas faster?
    :param behav_df:
    :return:
    '''

    return behav_df[['completed', 'rewarded', 'correct', 'stim_dir', 'resp_dir', 'DV', 'WT', 'trial', 'session', 'confidence', 'before_switch',
    'evidence', 'prior', 'probe',  'prev_completed_1', 'prev_evidence_1', 'prev_stim_dir_1', 'prev_rewarded_1', ''
                     'prev_resp_dir_1', 'prev_correct_1', 'WT_qs', 'prev_WT_qs_1']]

def convert_df(behav_df, session_type='SessionData', WTThresh=None, trim_last_trial=True):
    '''
    TODO: Map confidence for waiting time rats

    :param behav_df: input dataframe from bpod, psychphysics etc
    :param column_dict: dictionary containing keyvalue pairs that specify how to rename the behav_df
    :return: behav_df according to standard column naming convention. Required columns:
        TrialNumber - an increasing number that specifies in order the trials. One trial number per row of behavDF, this
                    will be the index
        DV - ranges from -1 to 1, the X axis of a psychometric function (assumes linear variation of difficulty)
        evidence - ranges from 0 to 1, often the absolute value of DV
        resp_dir - 0 or 1 - in standard 2AFC tasks 0 corresponds to the left port, 1 corresponds to the right port.
        correct - boolean True or False, specifies whether or not the animal made a correct choice on the current trial
        rewarded - boolean True or False, specifies whether or not the animal was rewarded on the current trial
                (often the same as correct, the exception is probe trials in the reward bias task)
        reward -  value of the recieved reward
    '''
    if session_type == 'RatPriors':
        print('WARNING: Assuming no probe trials, all completed rewarded')
        behav_df['CatchTrial'] = False
        behav_df['rewarded'] = behav_df['was_correct'].astype('bool')
        behav_df['correct'] = behav_df['was_correct'].astype('bool')
        behav_df['resp_dir'] = behav_df['resp_rightward']
        behav_df['DV'] = behav_df['signed_noise']
        behav_df['WT'] = .5
        behav_df['completed'] = behav_df['was_completed'].astype('bool')
        behav_df['trial'] = np.arange(0, len(behav_df))
        behav_df['before_switch'] = 0

    if session_type == 'SessionData':
        #behav_df = behav_df.rename(columns=column_dict)
        behav_df['rewarded'] = behav_df['Rewarded'].astype('bool')
        behav_df['success'] = 'Correct'
        behav_df.loc[behav_df['Rewarded'] == 0, 'success'] = 'Error'

        behav_df['correct'] = behav_df['CorrectChoice'].astype('bool')
        behav_df['completed'] = behav_df['CompletedTrial'].astype('bool')
        if "ChosenDirection" in behav_df.keys():
            behav_df["resp_dir"] = behav_df["ChosenDirection"] - 1
        else:
            behav_df['resp_dir'] = behav_df['ChosenDirectionBis'] - 1
        behav_df['response direction'] = 'Right'
        behav_df.loc[behav_df['resp_dir'] == 0, 'response direction'] = 'Left'
        behav_df['trial'] = behav_df['TrialNumber']
        behav_df['WT'] = behav_df['WaitingTime']


        behav_df.loc[:, 'next_trial_start'] = behav_df.loc[:, 'TrialStartAligned'].shift(-1).to_numpy()
        trial_len = (behav_df['next_trial_start'] - behav_df['TrialStartAligned']).to_numpy()
        behav_df['trial_len'] = trial_len

        # so that we can use this code for session data that doesnt have catch trials!
        if 'prior' not in behav_df.keys():
            behav_df['prior'] = .5
        if 'DV' not in behav_df.keys():
            behav_df['DV'] = behav_df['BetaDiscri']

        if 'before_switch' not in behav_df.keys():
            behav_df['before_switch'] = 0

    behav_df['stim_dir'] = np.sign(behav_df['DV'])  # sessiondata matlab records have 1= left, 2 = right
    behav_df.loc[:, 'stim_dir'] = behav_df['stim_dir'].replace({-1:0}).to_numpy()
    behav_df['stimulus direction'] = 'Right'
    behav_df.loc[behav_df['stim_dir'] == 0, 'stimulus direction'] = 'Left'

    behav_df['evidence'] = (behav_df['DV'] * 2).round(0) / 2

    behav_df = add_history_fields(behav_df, [1], ['completed', 'rewarded', 'evidence', 'stim_dir', 'WT', 'resp_dir', 'correct'])
    behav_df.loc[:, 'prev_completed_1'] = behav_df['prev_completed_1'].astype('bool')
    behav_df.loc[:, 'prev_correct_1'] = behav_df['prev_correct_1'].astype('bool')
    behav_df.loc[:, 'prev_rewarded_1'] = behav_df['prev_rewarded_1'].astype('bool')

    behav_df.loc[:, 'prev_correct_1'] = behav_df['prev_correct_1'].fillna(False)
    behav_df.loc[:, 'prev_completed_1'] = behav_df['prev_completed_1'].fillna(False)

    behav_df['probe'] = behav_df['CatchTrial']

    behav_df['confidence'] = behav_df['WT']



    if "ValidAlignedTrials" in behav_df.keys():
        behav_df  = behav_df[behav_df.completed & behav_df.ValidAlignedTrials]
        print("only using trials with valid alignment")
    else:
        behav_df = behav_df[behav_df['completed']]

    if WTThresh:
        behav_df = behav_df[behav_df.WaitingTime > WTThresh]

    if trim_last_trial:
        print("warning triming the last trial")
        behav_df = behav_df[behav_df.trial < max(behav_df.trial)]

    bins = np.quantile(behav_df.WT, [.33, .66])
    behav_df['WT_qs'] = np.digitize(behav_df.WT, bins)
    behav_df['prev_WT_qs_1'] = np.digitize(behav_df.prev_WT_1, bins)

    return behav_df.reset_index()

def run_psytrack(data, figure_folder):
    import psytrack as psy

    # put the behavioral data into a form that is useful for psytrack
    # we should have a dictionary with a key y and a key inputs

    # y is a 1D array of choices mapped to 1,2
    # inputs is another dict with arbitary keys, each containing a (num trials, M) array
    # where M is arbitrary but is usually about the previous i-trials

    dayLength = [np.sum(data['session'] == s) for s in np.unique(data['session'])]

    D = {'y': data['choice'].replace({0: 1, 1: 2}).to_numpy(), 'dayLength': np.array(dayLength),
         'answer': data['stimulus'].replace({-1: 1, 1: 2}).to_numpy(), 'correct': data['was_correct'].to_numpy()}

    # make an array were s[i,j] is the stimulus j trials back on the i-th trial
    # TODO change this to history of coherences and or history of responses
    num_hist = 5
    s_labels = ['DV', 'block']
    # s_labels = ['raw_signed_noise']
    for i in np.arange(1, num_hist + 1):
        s_labels.append('prev_choice_1' + str(i))

    s = data[s_labels]
    s['stim_dir'] = data['DV'].to_numpy()
    s['prior'] = data['blockNumber'].to_numpy()
    s = s.to_numpy()
    D['inputs'] = {'s1': s}

    print("The keys of the dict for this example animal:\n   ", list(D.keys()))
    print("The shape of y:   ", D['y'].shape)
    print("The number of trials:   N =", D['y'].shape[0])
    print("The unique entries of y:   ", np.unique(D['y']))
    print("The keys of inputs:\n   ", list(D['inputs'].keys()))
    print("\nThe shape of s1:", D['inputs']['s1'].shape)

    # Params for fitting
    weights = {'bias': 1,
               's1': 3}  # coherence, prior

    # It is often useful to have the total number of weights K in your model
    K = np.sum([weights[i] for i in weights.keys()])

    # Hyperparams
    hyper = {'sigInit': 2 ** 4.,  # Set to a single, large value for all weights. Will not be optimized further.
             'sigma': [2 ** -4.] * K,
             # Each weight will have it's own sigma optimized, but all are initialized the same
             'sigDay': 0.1}  # Indicates that session boundaries will be ignored in the optimization

    optList = ['sigma', 'sigDay']

    hyp, evd, wMode, hess_info = psy.hyperOpt(D, hyper, weights, optList)

    # fig = psy.plot_weights(wMode, weights)# %%

    prior_list = [np.max(data[data['session'] == i].prior) for i in np.unique(data.session)]

    sns.set_style('ticks')
    sns.set_context("paper", font_scale=2)
    # sns.set_palette("pastel")
    start_point = [0, -2]
    for i, val in enumerate(dayLength):
        rectangle = plt.Rectangle(start_point, val, 8, fc=[1 - prior_list[i], 1 - prior_list[i], prior_list[i]],
                                  alpha=0.1)
        plt.gca().add_patch(rectangle)
        start_point = [np.cumsum(dayLength)[i], -2]

    plt.plot(wMode.T, label=['bias', 'stimulus', 'prior', 'previous response']);
    plt.legend();
    plt.plot([0, D['y'].shape[0]], [0, 0], 'k--')
    [plt.plot([d, d], [-2, 6], 'k--', alpha=0.2) for d in np.cumsum(dayLength)];
    plt.show();


def add_history_fields(behav_df, shifts, history_keys):
    for s in shifts:
        for key in history_keys:
            new_key = 'prev_' + key + '_' + str(s)
            behav_df[new_key] = behav_df[key].shift(s)
    return behav_df


def condition_sim(SIM):
    SIM["evidence"] = ((SIM["stim_external"] * 4).round(0) / 4).round(1).to_numpy()
    SIM["prev_evidence_1"] = SIM["evidence"].shift(1)
    SIM["resp_dir"] = SIM["decision"]
    SIM["prev_resp_dir_1"] = SIM["resp_dir"].shift(1)
    SIM["prev_correct_1"] = SIM["correct"].shift(1)
    SIM["DV"] = SIM["stim_external"]
    SIM["correct"] = SIM["correct"].astype("bool")
    SIM["prev_completed_1"] = True
    return SIM

def condition_behavdf(data):
    data["evidence"] = ((data["raw_signed_noise"] * 3).round(0) / 3).round(1).to_numpy()
    data["prev_evidence"] = data["evidence"].shift(1).to_numpy()
    data["prev_was_correct"] = data["prev_was_correct"].astype('bool')
    data["prev_resp_rightward1"] = data["resp_rightward"].shift(1).to_numpy()
    data["DV"] = data["raw_signed_noise"].to_numpy()
    data["confidence"] = 0
    data["correct"] = data["was_correct"].astype("bool")
    return data

def map_p_correct(behav_df):
    behav_df['p_correct'] = behav_df.groupby(behav_df['DV'].round(1).abs()).transform(np.nanmean)["correct"]
    return behav_df
