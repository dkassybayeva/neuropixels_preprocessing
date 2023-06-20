import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from joblib import load, dump
import matplotlib.pyplot as plt


def calc_event_outcomes(output_dir, metadata):
    """
    Creates additional useful fields in the session data (trial events).

    :param output_dir: [string] directory where TrialEvents.npy was saved.
    :return: [None] Overwrites TrialEvents.npy
    """
    OTT_LAB_DATA = metadata['ott_lab']

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

    if OTT_LAB_DATA:
        _sd_custom = _sd['Custom']['TrialData']
    else:
        _sd_custom = _sd['Custom']

    # Chosen direction (1=left, 2=right, -1=nan)
    choice_left = _sd_custom['ChoiceLeft'][:n_trials]
    left_choice_idx, right_choice_idx = one_zero_idx(choice_left)
    _sd['ChosenDirection'] = -1 * np.ones(n_trials)
    _sd['ChosenDirection'][left_choice_idx] = 1
    _sd['ChosenDirection'][right_choice_idx] = 2

    if not metadata['task']=='matching':
        # Correct and error trials
        choice_correct = _sd_custom['ChoiceCorrect'][:n_trials]
        _sd['CorrectChoice'], _sd['PunishedTrial'] = one_zero_idx(choice_correct)

    # Trial where rat gave a response
    _sd['CompletedTrial'] = (choice_left > -1) & (_sd['TrialNumber'] > 30)

    # Rewarded Trials
    _sd['Rewarded'], no_reward = one_zero_idx(_sd_custom['Rewarded'][:n_trials])

    # Trials where rat sampled but did not respond
    complete, incomplete = one_zero_idx(_sd['CompletedTrial'])
    early_withdrawal = _sd_custom['EarlyWithdrawal'][:n_trials] == 1
    _sd['UnansweredTrials'] = incomplete & early_withdrawal
    assert _sd['UnansweredTrials'].sum() <= min(early_withdrawal.sum(), incomplete.sum())

    if not metadata['task'] == 'matching':
        # CatchTrial
        _sd['CatchTrial'] = _sd_custom['CatchTrial'][:n_trials]

        # Correct catch trials
        _sd['CompletedCatchTrial'] = complete & (_sd['CatchTrial'] == 1)

        # Correct trials, but rat was waiting too short
        wait_too_short = _sd_custom['FeedbackTime'][:n_trials] < 0.5
        _sd['CorrectShortWTTrial'] = _sd['CorrectChoice'] & wait_too_short

        # These are all the waiting time trials (correct catch and incorrect trials)
        catch_or_wrong = _sd['CompletedCatchTrial'] & _sd['PunishedTrial']
        _sd['CompletedWTTrial'] = catch_or_wrong & complete
        assert _sd['CompletedWTTrial'].sum() <= min(catch_or_wrong.sum(), complete.sum())

        # Trials where rat answered but did not receive reward
        punish_or_catch = _sd['PunishedTrial'] | _sd['CatchTrial']
        _sd['WaitingTimeTrial'] = punish_or_catch & complete

        _sd['WaitingTime'] = _sd_custom['FeedbackTime'][:n_trials]
    else:
        _sd['WaitingTime'] = _sd_custom['FeedbackWaitingTime'][:n_trials]

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
        if nt_reward and nt_mod == 1:
            code = 1
        elif nt_reward and nt_mod == 2:
            code = 2
        elif ~nt_reward and nt_complete and nt_mod == 1:
            code = 3
        elif ~nt_reward and nt_complete and nt_mod == 2:
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

        if nt_mod == 1 and nt_chosen_dir == 1 and nt_complete:
            code2 = 1
        elif nt_mod == 2 and nt_chosen_dir == 1 and nt_complete:
            code2 = 2
        elif nt_mod == 1 and nt_chosen_dir == 2 and nt_complete:
            code2 = 3
        elif nt_mod == 2 and nt_chosen_dir == 2 and nt_complete:
            code2 = 4
        else:
            code2 = np.nan
        _sd['ModReward'][nt] = code2

    # Waiting time split
    _sd['WaitingTimeSplit'] = np.full(_sd['ChosenDirection'].shape, np.nan)

    complete_no_reward = complete & no_reward

    long_idx = _sd['WaitingTime'] >= 6.5
    midlong_idx = (_sd['WaitingTime'] < 6.5) & (_sd['WaitingTime'] >= 5.5)
    midshort_idx = (_sd['WaitingTime'] < 5.5) & (_sd['WaitingTime'] >= 4)
    short_idx = (_sd['WaitingTime'] < 4) & (_sd['WaitingTime'] >= 2.5)
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

    if metadata['task'] == 'matching':
        Cin_str = 'StartCIn'
        StimOn_str = 'Sampling'  # For matching, this variable is nonsensical
        Rin_str = 'StartRIn'
        Lin_str = 'StartLIn'
        Feedback_str = 'FeedbackWaitingTime'  # Time spent in the choice port before leaving or reward
        Sample_str = 'SampleTime'
        # DV_str = 'DecisionVariable'
    else:
        Cin_str = 'stay_Cin'
        StimOn_str = 'stimulus_delivery_min'
        Rin_str = 'start_Rin'
        Lin_str = 'start_Lin'
        Feedback_str = 'FeedbackTime'
        Sample_str = 'SampleLength' if OTT_LAB_DATA else 'ST'
        DV_str = 'DecisionVariable' if OTT_LAB_DATA else 'DV'

    for nt in range(n_trials):
        nt_states = _sd['RawEvents']['Trial'][nt]['States']

        _sd['PokeCenterStart'][nt] = nt_states[Cin_str][0]
        _sd['StimulusOnset'][nt] = nt_states[StimOn_str][0]

        if ~np.isnan(nt_states[Rin_str][0]):
            _sd['ResponseStart'][nt] = nt_states[Rin_str][0]
            _sd['ResponseEnd'][nt] = nt_states[Rin_str][0] + _sd_custom[Feedback_str][nt]
        elif ~np.isnan(nt_states[Lin_str][0]):
            _sd['ResponseStart'][nt] = nt_states[Lin_str][0]
            _sd['ResponseEnd'][nt] = nt_states[Lin_str][0] + _sd_custom[Feedback_str][nt]
        else:
            _sd['ResponseStart'][nt] = np.nan
            _sd['ResponseEnd'][nt] = np.nan

        if not OTT_LAB_DATA:
            nt_GUI = _sd['TrialSettings'][nt]['GUI']
            if 'LaserTrials' in nt_GUI.keys():
                if nt_GUI['LaserTrials'] > 0:
                    if 'LaserTrainDuration_ms' in nt_GUI.keys():
                        _sd['LaserTrialTrainLength'][nt] = nt_GUI['LaserTrainDuration_ms']
                    else:  # old version
                        _sd['LaserTrialTrainLength'][nt] = np.nan
                else:
                    _sd['LaserTrialTrainLength'][nt] = np.nan
            else:  # not even Laser Trials settings, very old version
                _sd['LaserTrialTrainLength'][nt] = np.nan

    _sd['SamplingDuration'] = _sd_custom[Sample_str][:n_trials]
    _sd['StimulusOffset'] = _sd['StimulusOnset'] + _sd['SamplingDuration']

    _sd['ChosenDirectionBis'] = _sd['ChosenDirection']
    _sd['ChosenDirectionBis'][_sd['ChosenDirectionBis'] == 3] = np.nan

    # Correct length of TrialStartAligned
    _sd['TrialStartTimestamp'] = _sd['TrialStartTimestamp'][:n_trials]
    if 'TrialEndTimestamp' in _sd.keys():
        _sd['TrialEndTimestamp'] = _sd['TrialEndTimestamp'][:n_trials]
    if 'TrialTypes' in _sd.keys():
        _sd['TrialTypes'] = _sd['TrialTypes'][:n_trials]
    _sd['TrialSettings'] = _sd['TrialSettings'][:n_trials]

    if OTT_LAB_DATA:
        _sd['recorded_TTL_trial_start_time'] = _sd['recorded_TTL_trial_start_time'][:n_trials]
        _sd['no_matching_TTL_start_time'] = _sd['no_matching_TTL_start_time'][:n_trials]
        _sd['large_TTL_gap_after_start'] = _sd['large_TTL_gap_after_start'][:n_trials]
    else:
        _sd['TrialStartAligned'] = _sd['TrialStartAligned'][:n_trials]

        # laser trials
        if 'LaserTrial' in _sd_custom.keys() and _sd_custom['LaserTrial'].sum() > 0:
            if 'LaserTrialTrainStart' in _sd_custom.keys():
                _sd['LaserTrialTrainStart'] = _sd_custom['LaserTrialTrainStart'][:n_trials]
                _sd['LaserTrialTrainStartAbs'] = _sd['LaserTrialTrainStart'] + _sd['ResponseStart']
                _sd['LaserTrial'] = _sd_custom['LaserTrial'][:n_trials]
                _sd['LaserTrial'][_sd['CompletedTrial'] == 0] = 0
                _sd['LaserTrial'][_sd['LaserTrialTrainStartAbs'] > _sd['ResponseEnd']] = 0
                _sd['LaserTrialTrainStartAbs'][_sd['LaserTrial'] != 1] = np.nan
                _sd['LaserTrialTrainStart'][_sd['LaserTrial'] != 1] = np.nan

                _sd['CompletedWTLaserTrial'] = _sd['LaserTrial']
                _sd['CompletedWTLaserTrial'][_sd['CompletedWTTrial'] != 1] = np.nan
            else:  # old version, laser during entire time investment
                _sd['LaserTrialTrainStart'] = np.zeros(n_trials)
                _sd['LaserTrialTrainStartAbs'] = _sd['ResponseStart']
                _sd['LaserTrial'] = _sd_custom['LaserTrial'][:n_trials]
                _sd['LaserTrial'][_sd['CompletedTrial'] == 0] = 0
                _sd['LaserTrialTrainStartAbs'][_sd['LaserTrial'] != 1] = np.nan
                _sd['LaserTrialTrainStart'][_sd['LaserTrial'] != 1] = np.nan
        else:
            _sd['LaserTrialTrainStart'] = np.full(n_trials, np.nan)
            _sd['LaserTrialTrainStartAbs'] = np.full(n_trials, np.nan)
            _sd['LaserTrial'] = np.zeros(n_trials)
            _sd['CompletedWTLaserTrial'] = np.full(n_trials, np.nan)
            _sd['CompletedWTLaserTrial'][_sd['CompletedWTTrial'] == 1] = 0

    if 'BlockNumber' in _sd_custom.keys():
        _sd['BlockNumber'] = _sd_custom['BlockNumber'][:n_trials]

    if not OTT_LAB_DATA:
        if 'RewardMagnitude' in _sd_custom.keys():
            _sd['RelativeReward'] = _sd_custom['RewardMagnitude'][:, 0] - _sd_custom['RewardMagnitude'][:, 1]
            _sd['RelativeReward'] = _sd['RelativeReward'][:n_trials]

    # discrimination measures
    if 'AuditoryOmega' in _sd_custom.keys():
        _sd['DV'] = _sd_custom[DV_str][:n_trials]
        _sd['MostClickSide'] = -1 * np.ones(n_trials)
        _sd['OmegaDiscri'] = 2 * np.abs(_sd_custom['AuditoryOmega'][:n_trials] - 0.5)
        _sd['NRightClicks'] = np.zeros(n_trials)
        _sd['NLeftClicks'] = np.zeros(n_trials)
        for trial_i in range(n_trials):
            rct = _sd_custom['RightClickTrain'][trial_i]
            if type(rct) == np.ndarray:
                _sd['NRightClicks'][trial_i] = len(rct)
            else:
                _sd['NRightClicks'][trial_i] = 1

            lct = _sd_custom['LeftClickTrain'][trial_i]
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
    # else:
    #     _sd['ChoiceGivenClick'] = _sd['CorrectChoice'][:n_trials]


    if not OTT_LAB_DATA:
        if 'RewardMagnitude' in _sd_custom.keys():
            _sd['RewardMagnitude'] = _sd_custom['RewardMagnitude'][:n_trials, :]

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
    elif not metadata['task'] == 'matching':
        if 'RewardMagnitudeL' in _sd_custom.keys():
            _sd['RewardMagnitudeL'] = _sd_custom['RewardMagnitudeL'][:n_trials]
        if 'RewardMagnitudeR' in _sd_custom.keys():
            _sd['RewardMagnitudeR'] = _sd_custom['RewardMagnitudeR'][:n_trials]

        # Conditioning the trials
        _sd['RewardMagnitudeCorrect'] = -1 * np.ones(n_trials)
        # Get magnitudes for left choices
        left_choices = _sd['ChosenDirection'] == 1
        _sd['RewardMagnitudeCorrect'][left_choices] = _sd['RewardMagnitudeL'][left_choices]
        assert n_trials - np.sum(_sd['RewardMagnitudeCorrect'] == -1) == left_choices.sum()

        right_choices = _sd['ChosenDirection'] == 2
        _sd['RewardMagnitudeCorrect'][right_choices] = _sd['RewardMagnitudeR'][right_choices]
        assert n_trials - np.sum(_sd['RewardMagnitudeCorrect'] == -1) == left_choices.sum() + right_choices.sum()

    dump(_sd, output_dir + 'TrialEvents.npy', compress=3)

    print('Trial event conditions/outcomes calculated and added to TrialEvents.npy.')


def create_behavioral_dataframe(output_dir):
    """
    Create simpler copy of Trial Events to turn into pandas dataframe
    """
    print('Creating behavioral dataframe')
    print('Removing fields that are not trialwise...', end='')
    # Load the session data in TrialEvents.npy
    _sd = load(output_dir + 'TrialEvents.npy')

    # Remove keys no longer needed for spiking data alignment
    n_keys_start = len(_sd.keys())

    non_trialwise_items = ['Custom', 'RawEvents', 'nTrials', 'TrialSettings', 'Settings',
                           'Info', 'SettingsFile', 'RawData', 'RewardMagnitude', 'CompletedChosenDirection',
                           'CompletedCatchTrial', 'ModReward', 'SideReward', 'ChosenDirectionBis',
                           'CorrectShortWTTrial', 'CompletedWTLaserTrial', 'CompletedWTTrial', 'PunishedTrial']

    for ntw_item in non_trialwise_items:
        if ntw_item in _sd.keys():
            _sd.pop(ntw_item)

    print(f"{n_keys_start - len(_sd.keys())} fields removed.")

    # Convert to dataframe and save
    behav_dict = {key: np.array(_sd[key]).squeeze() for key in _sd if '__' not in key and key != 'Settings'}
    behav_df = pd.DataFrame.from_dict(behav_dict)

    # so that we can use this code for session data that doesnt have catch trials!
    if 'CatchTrial' in behav_df.keys():
        behav_df.CatchTrial = behav_df.CatchTrial.astype('bool')
    if 'CompletedTrial' in behav_df.keys():
        behav_df.CompletedTrial = behav_df.CompletedTrial.astype('bool')

    dump(behav_df, output_dir + "behav_df", compress=3)

    print('Behavioral dataframe saved to: ' + output_dir + "behav_df")


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

def convert_df(behav_df, metadata, session_type='SessionData', trim_last_trial=True):
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
    if metadata['ott_lab']:
        WTThresh = False if metadata['task'] == 'reward-bias' else True
    else:
        WTThresh = metadata['time_investment']

    discrimination_task = False if metadata['task'] == 'matching' else True

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

        if 'CorrectChoice' in behav_df.keys():
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

        if metadata['ott_lab']:
            trialstart_str = 'recorded_TTL_trial_start_time'
        else:
            trialstart_str = 'TrialStartAligned'
        behav_df.loc[:, 'next_trial_start'] = behav_df.loc[:, trialstart_str].shift(-1).to_numpy()
        trial_len = (behav_df['next_trial_start'] - behav_df[trialstart_str]).to_numpy()
        behav_df['trial_len'] = trial_len

        # so that we can use this code for session data that doesnt have catch trials!
        if 'prior' not in behav_df.keys():
            behav_df['prior'] = .5


        if 'before_switch' not in behav_df.keys():
            behav_df['before_switch'] = 0

    if discrimination_task:
        if 'DV' not in behav_df.keys():
            behav_df['DV'] = behav_df['BetaDiscri']
        behav_df['stim_dir'] = np.sign(behav_df['DV'])  # sessiondata matlab records have 1= left, 2 = right
        behav_df.loc[:, 'stim_dir'] = behav_df['stim_dir'].replace({-1:0}).to_numpy()
        behav_df['stimulus direction'] = 'Right'
        behav_df.loc[behav_df['stim_dir'] == 0, 'stimulus direction'] = 'Left'

        behav_df['evidence'] = (behav_df['DV'] * 2).round(0) / 2

        if not metadata['ott_lab']:
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
        if metadata['ott_lab']:
            behav_df = behav_df[~behav_df['no_matching_TTL_start_time']]
            behav_df = behav_df[behav_df['large_TTL_gap_after_start']==0]
        behav_df = behav_df[behav_df['completed']]

    if not metadata['ott_lab']:
        if WTThresh:
            behav_df = behav_df[behav_df.WaitingTime > WTThresh]

    if trim_last_trial:
        print("warning triming the last trial")
        behav_df = behav_df[behav_df.trial < max(behav_df.trial)]

    bins = np.quantile(behav_df.WT, [.33, .66])
    behav_df['WT_qs'] = np.digitize(behav_df.WT, bins)

    if not metadata['ott_lab']:
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
