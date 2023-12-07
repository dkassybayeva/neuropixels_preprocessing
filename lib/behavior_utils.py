import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from joblib import load, dump
import matplotlib.pyplot as plt


def filter_valid_time_investment_trials(behav_data, minimum_wait_time=2.0):
    """

    :param behav_data:
    :param minimum_wait_time: [seconds] Lower cut-off for a valid 'waiting-time'
    :return:
    """
    # CatchTrial
    assert ~np.any([x in np.where(behav_data['Rewarded'])[0] for x in np.where(behav_data['CatchTrial'])[0]])
    assert ~np.any([x in np.where(behav_data['Rewarded'])[0] for x in np.where(behav_data['ErrorChoice'])[0]])

    # Correct catch trials (unrewarded -> the animal had to leave of its own volition)
    behav_data['CorrectCatch'] = behav_data['CorrectChoice'] & behav_data['CatchTrial']
    behav_data['SkippedReward'] = (behav_data['CorrectChoice'] & ~behav_data['CatchTrial']) & \
                                  (behav_data['CorrectChoice'] & ~behav_data['Rewarded'])

    assert behav_data['CorrectCatch'].sum() + behav_data['SkippedReward'].sum() + behav_data['Rewarded'].sum() == \
           behav_data['CorrectChoice'].sum()

    # These are all the waiting time trials (correct catch and incorrect trials)
    behav_data['ChoiceNoFeedback'] = behav_data['CorrectCatch'] | behav_data['ErrorChoice'] | behav_data['SkippedReward']
    assert behav_data['ChoiceNoFeedback'].sum() + behav_data['Rewarded'].sum() == behav_data['MadeChoice'].sum()

    # Valid waiting-time trials are those in which the rat made a choice, but received no Feedback from the system
    # e.g., no reward and no indication that the trial ended (handled by setting Catch or Error time-ups to 20s).
    behav_data['WaitingTimeTrial'] = behav_data['ChoiceNoFeedback'] & (behav_data['WaitingTime'] > minimum_wait_time)

    return behav_data


def separate_waiting_durations(_sd):
    _sd['WaitingTimeSplit'] = np.full(_sd['WaitingTime'].shape, "", dtype=object)
    long_idx = _sd['WaitingTime'] >= 6.5
    midlong_idx = (_sd['WaitingTime'] < 6.5) & (_sd['WaitingTime'] >= 5.5)
    midshort_idx = (_sd['WaitingTime'] < 5.5) & (_sd['WaitingTime'] >= 4)
    short_idx = (_sd['WaitingTime'] < 4) & (_sd['WaitingTime'] >= 2)
    assert ~np.any([x in np.where(long_idx)[0] for x in np.where(midlong_idx)[0]])
    assert ~np.any([x in np.where(midlong_idx)[0] for x in np.where(midshort_idx)[0]])
    assert ~np.any([x in np.where(midshort_idx)[0] for x in np.where(short_idx)[0]])

    _sd.loc[(_sd['WaitingTimeTrial'] & short_idx), 'WaitingTimeSplit'] = 'short'
    _sd.loc[(_sd['WaitingTimeTrial'] & midshort_idx), 'WaitingTimeSplit'] = 'mid_short'
    _sd.loc[(_sd['WaitingTimeTrial'] & midlong_idx), 'WaitingTimeSplit'] = 'mid_long'
    _sd.loc[(_sd['WaitingTimeTrial'] & long_idx), 'WaitingTimeSplit'] = 'long'
    return _sd


def calc_event_outcomes(behav_data, metadata):
    """
    Creates additional useful fields in the session data (trial events).

    :param output_dir: [string] directory where TrialEvents.npy was saved.
    :return: [None] Overwrites TrialEvents.npy
    """
    def one_zero_idx(data_obj):
        one_choice_idx = data_obj == 1.
        zero_choice_idx = data_obj == 0.
        return one_choice_idx, zero_choice_idx

    # --------------------------------------------------------------------- #
    #                       Get input data
    # --------------------------------------------------------------------- #
    OTT_LAB_DATA = metadata['ott_lab']
    if OTT_LAB_DATA:
        _sd_custom = behav_data['Custom']['TrialData']
    else:
        _sd_custom = behav_data['Custom']
    # --------------------------------------------------------------------- #


    # --------------------------------------------------------------------- #
    #  new session dataframe for trialwise data in which we are interested
    # --------------------------------------------------------------------- #
    n_trials = behav_data['nTrials']
    _sd = pd.DataFrame()
    _sd['TrialNumber'] = np.arange(n_trials)
    # --------------------------------------------------------------------- #


    # --------------------------------------------------------------------- #
    #                       Choice variables
    # --------------------------------------------------------------------- #
    choice_left = _sd_custom['ChoiceLeft'][:n_trials]
    left_choice_idx, right_choice_idx = one_zero_idx(choice_left)
    _sd['ChoseLeft'] = left_choice_idx
    _sd['ChoseRight'] = right_choice_idx
    _sd['MadeChoice'] = left_choice_idx | right_choice_idx
    _sd['NoChoice'] = np.isnan(choice_left)
    _sd['ChoiceSide'] = ''
    _sd.loc[_sd.ChoseLeft, 'ChoiceSide'] = 'Left'
    _sd.loc[_sd.ChoseRight, 'ChoiceSide'] = 'Right'

    assert np.all((_sd['ChoseLeft'] | _sd['ChoseRight']) == ~_sd['NoChoice'])
    assert np.all(_sd['MadeChoice'] == ~_sd['NoChoice'])

    if not metadata['task']=='matching':
        # Correct and error trials
        choice_correct = _sd_custom['ChoiceCorrect'][:n_trials]
        _sd['CorrectChoice'], _sd['ErrorChoice'] = one_zero_idx(choice_correct)
        assert np.all((_sd['CorrectChoice'] | _sd['ErrorChoice']) == ~_sd['NoChoice'])
        assert ~np.any([x in np.where(_sd['ErrorChoice'])[0] for x in np.where(_sd['CorrectChoice'])[0]])
    # --------------------------------------------------------------------- #


    # ---------------------------------------------------------------------- #
    #                               Reward
    # ---------------------------------------------------------------------- #
    _sd['Rewarded'], no_reward = one_zero_idx(_sd_custom['Rewarded'][:n_trials])
    if OTT_LAB_DATA:
        _sd['RewardMagnitudeL'] = _sd_custom['RewardMagnitudeL'][:n_trials]
        _sd['RewardMagnitudeR'] = _sd_custom['RewardMagnitudeR'][:n_trials]
    else:
        if 'RewardMagnitude' in _sd_custom.keys():
            _sd['RewardMagnitudeL'] = _sd_custom['RewardMagnitude'][:n_trials, 0].astype('int')
            _sd['RewardMagnitudeR'] = _sd_custom['RewardMagnitude'][:n_trials, 1].astype('int')
    _sd['RewardMagnitudeCorrect'] = np.full(n_trials, np.nan)
    _sd.loc[(_sd['ChoseLeft'] & _sd['CorrectChoice']), 'RewardMagnitudeCorrect'] = _sd['RewardMagnitudeL'][_sd['ChoseLeft'] & _sd['CorrectChoice']]
    _sd.loc[(_sd['ChoseRight'] & _sd['CorrectChoice']), 'RewardMagnitudeCorrect'] = _sd['RewardMagnitudeR'][_sd['ChoseRight'] & _sd['CorrectChoice']]
    assert np.isnan(_sd['RewardMagnitudeCorrect']).sum() == (_sd['ErrorChoice'].sum() + _sd['NoChoice'].sum())
    _sd['RelativeReward'] = _sd['RewardMagnitudeL'] - _sd['RewardMagnitudeR']
    # --------------------------------------------------------------------- #


    # --------------------------------------------------------------------- #
    #                         Waiting Time
    # --------------------------------------------------------------------- #
    _sd['EarlyWithdrawal'] = _sd_custom['EarlyWithdrawal'][:n_trials] == 1
    if metadata['task'] == 'matching':
        _sd['WaitingTime'] = _sd_custom['FeedbackWaitingTime'][:n_trials]
    else:
        _sd['CatchTrial'] = _sd_custom['CatchTrial'][:n_trials] == 1
        _sd['WaitingTime'] = _sd_custom['FeedbackTime'][:n_trials]
        _sd = filter_valid_time_investment_trials(_sd)

    _sd = separate_waiting_durations(_sd)
    # --------------------------------------------------------------------- #


    # ------------------------------------------------------------------------ #
    #                       Trial timestamps
    # ------------------------------------------------------------------------ #
    _sd['StimulusOnset'] = np.zeros(n_trials)
    _sd['PokeCenterStart'] = np.zeros(n_trials)

    _sd['ResponseStart'] = np.zeros(n_trials)
    _sd['ResponseEnd'] = np.zeros(n_trials)

    if metadata['task'] == 'matching':
        Cin_str = 'StartCIn'
        StimOn_str = 'Sampling'  # For matching, this variable is nonsensical
        Rin_str = 'StartRIn'
        Lin_str = 'StartLIn'
        Feedback_str = 'FeedbackWaitingTime'  # Time spent in the choice port before leaving or reward
        Sample_str = 'SampleTime'
    else:
        Cin_str = 'stay_Cin'
        StimOn_str = 'stimulus_delivery_min'
        Rin_str = 'start_Rin'
        Lin_str = 'start_Lin'
        Feedback_str = 'FeedbackTime'
        Sample_str = 'SampleLength' if OTT_LAB_DATA else 'ST'
        DV_str = 'DecisionVariable' if OTT_LAB_DATA else 'DV'
    # ------------------------------------------------------------------------ #

    for nt in range(n_trials):
        nt_states = behav_data['RawEvents']['Trial'][nt]['States']

        _sd.loc[nt, 'PokeCenterStart'] = nt_states[Cin_str][0]
        _sd.loc[nt, 'StimulusOnset'] = nt_states[StimOn_str][0]

        if ~np.isnan(nt_states[Rin_str][0]):
            _sd.loc[nt,'ResponseStart'] = nt_states[Rin_str][0]
            _sd.loc[nt,'ResponseEnd'] = nt_states[Rin_str][0] + _sd_custom[Feedback_str][nt]
        elif ~np.isnan(nt_states[Lin_str][0]):
            _sd.loc[nt,'ResponseStart'] = nt_states[Lin_str][0]
            _sd.loc[nt,'ResponseEnd'] = nt_states[Lin_str][0] + _sd_custom[Feedback_str][nt]
        else:
            _sd.loc[nt,'ResponseStart'] = np.nan
            _sd.loc[nt,'ResponseEnd'] = np.nan

    _sd['SamplingDuration'] = _sd_custom[Sample_str][:n_trials]
    _sd['StimulusOffset'] = _sd['StimulusOnset'] + _sd['SamplingDuration']
    # ------------------------------------------------------------------------ #

    _sd['TrialStartTimestamp'] = behav_data['TrialStartTimestamp'][:n_trials]
    if 'TrialEndTimestamp' in _sd.keys():
        _sd['TrialEndTimestamp'] = behav_data['TrialEndTimestamp'][:n_trials]
    if 'TrialTypes' in _sd.keys():
        _sd['TrialTypes'] = behav_data['TrialTypes'][:n_trials]

    if OTT_LAB_DATA:
        _sd['TTLTrialStartTime'] = behav_data['recorded_TTL_trial_start_time'][:n_trials]
        _sd['no_matching_TTL_start_time'] = behav_data['no_matching_TTL_start_time'][:n_trials]
        _sd['large_TTL_gap_after_start'] = behav_data['large_TTL_gap_after_start'][:n_trials]

    else:
        _sd['TTLTrialStartTime'] = behav_data['TrialStartAligned'][:n_trials]

    _sd['NextTrialStart'] = _sd['TTLTrialStartTime'].shift(-1).to_numpy()
    _sd['TrialLength'] = (_sd['NextTrialStart'] - _sd['TTLTrialStartTime']).to_numpy()
    # ------------------------------------------------------------------------ #


    # ---------------------------------------------------------------------- #
    #                        Block block (Ha ha)
    # ---------------------------------------------------------------------- #
    if 'BlockNumber' in _sd_custom.keys():
        _sd['BlockNumber'] = _sd_custom['BlockNumber'][:n_trials]
    # ---------------------------------------------------------------------- #


    # ---------------------------------------------------------------------- #
    #                       Discrimination measures
    # ---------------------------------------------------------------------- #
    if DV_str in _sd_custom.keys():
        _sd['DV'] = _sd_custom[DV_str][:n_trials]
        _sd['NRightClicks'] = np.zeros(n_trials)
        _sd['NLeftClicks'] = np.zeros(n_trials)
        for trial_i in range(n_trials):
            rct = _sd_custom['RightClickTrain'][trial_i]
            if type(rct) == np.ndarray:
                _sd.loc[trial_i, 'NRightClicks'] = len(rct)
            else:
                _sd.loc[trial_i, 'NRightClicks'] = 1

            lct = _sd_custom['LeftClickTrain'][trial_i]
            if type(lct) == np.ndarray:
                _sd.loc[trial_i, 'NLeftClicks'] = len(lct)
            else:
                _sd.loc[trial_i, 'NLeftClicks'] = 1
        assert np.all(_sd['DV'] == (_sd['NLeftClicks'] - _sd['NRightClicks']) / (_sd['NLeftClicks'] + _sd['NRightClicks']))
        _sd['RatioDiscri'] = np.log10(_sd['NRightClicks'] / _sd['NLeftClicks'])
    # ---------------------------------------------------------------------- #

    return _sd


def create_behavioral_dataframe(output_dir, metadata):
    """
    Wrapper for calc_event_outcomes, which, given a behavioral dictionary
    returns a dataframe with variables of interest.

    This function simply loads the behavioral dictionary and saves the dataframe.
    """
    event_dict = load(output_dir + 'TrialEvents.npy')
    behav_df = calc_event_outcomes(event_dict, metadata)
    dump(behav_df, output_dir + "behav_df", compress=3)
    print('Behavioral dataframe saved to: ' + output_dir + "behav_df")




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
