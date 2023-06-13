from os import path, makedirs

import pandas as pd

# ----------------------------------------------------------------------- #
#                           Constants
# ----------------------------------------------------------------------- #
# -------recording--------- #
fs = 30e3  # Trodes sampling frequency in Hz
ms_converter = 1000 / fs
n_Npix1_electrodes = 960
n_active_electrodes = 384

# -------analysis bins--------- #
max_ISI = 0.001  # max intersample interval (ISI), above which the period is considered a "gap" in the recording
trace_subsample_bin_size_ms = 25  # sample period in ms
sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces
# ----------------------------------------------------------------------- #


def save_directory_helper():
    try:
        DATA_DIR = f'/home/mud/Workspace/ott_neuropix_data/'
        assert path.exists(DATA_DIR)
    except:
        DATA_DIR = 'D:NeuroData/'
        assert path.exists(DATA_DIR)
    return DATA_DIR


def write_session_metadata_to_csv():
    columns = ['ott_lab', 'rat_name', 'date', 'trodes_datetime', 'task', 'behav_datetime',
               'region', 'probe_num', 'trodes_config', 'recording_type', 'DIO_port_num',
               'time_investment', 'reward_bias', 'behavior_mat_file',
               'sps', 'kilosort_ver', 'experimenter',
               'linking_group', 'recording_session_id', 'prior', 'experiment_id', 'stimulus', 'behavior_phase']

    metadata = {'ott_lab': True,
                'rat_name': '1',
                'date': '20230507_123146',
                'trodes_datetime': '20230507_123146',
                'task': 'time-investment', #'matching'
                'behav_datetime': '20230507_122316',
                'probe_num': '1',
                'trodes_config': '2023-04-10.trodesconf',
                'recording_type': 'neuropixels_1.0',
                'DIO_port_num': 1,
                'time_investment': False,
                'reward_bias': False,
                'prior': False,  # Could possibly be Amy's code for a task type that was previously used
                'experiment_id': 'learning_uncertainty',
                'linking_group': None, #'Nina2',
                'recording_session_id': 0,
                'kilosort_ver': 2.5,
                'sps': sps
    }

    metadata['behavior_mat_file'] = f'{metadata["rat_name"]}_DiscriminationConfidence_{metadata["behav_datetime"]}.mat'
    # metadata['behavior_mat_file'] = f'{metadata["rat_name"]}_TwoArmBanditVariant_{metadata["behav_datetime"]}.mat',
    metadata['experimenter'] = 'Gregory Knoll' if metadata['ott_lab'] else 'Amy'
    metadata['region'] = 'lmPFC' if metadata['ott_lab'] else 'lOFC'
    # -----------------------------------#
    # List of stimuli for each experiment:
    #     'freq' =
    #     'freq_nat' =
    #     'nat' =
    #     'nat_nat' =
    # -----------------------------------#
    metadata['stimulus'] = 'freq'

    # -----------------------------------#
    # Session number code:
    #     -1 = good performance, no noise (or rare)
    #      0 = first day with noise
    #     -2 = poor performance, no noise (<70%)
    # -----------------------------------#
    metadata['behavior_phase'] = -1

    DATA_DIR = save_directory_helper()

    ephys_metadata_file = DATA_DIR + 'ephys_sessions_metadata.csv'
    try:
        ephys_df = pd.read_csv(ephys_metadata_file)
    except:
        ephys_df = pd.DataFrame(columns=columns)

    ephys_df.loc[len(ephys_df)] = metadata
    ephys_df.to_csv(ephys_metadata_file, index=False)

    return metadata


def load_session_metadata_from_csv(rat, session_date):
    DATA_DIR = save_directory_helper()
    ephys_metadata_file = DATA_DIR + 'ephys_sessions_metadata.csv'
    ephys_df = pd.read_csv(ephys_metadata_file)

    rat_idx = ephys_df['rat_name'].apply(str) == rat
    date_idx = ephys_df['date'].apply(str) == session_date
    return ephys_df[rat_idx & date_idx].iloc[0].to_dict()


def get_session_path(metadata):
    rat = metadata['rat_name']
    session = metadata['trodes_datetime']

    try:
        rec_dir = behav_dir = f'D:NeuroData/R{rat}/{session}.rec/'
        assert path.exists(rec_dir)
    except:
        rec_dir = f'/media/ottlab/data/{rat}/ephys/{session}.rec/'
        behav_dir = f'/media/ottlab/data/{rat}/bpod_session/{metadata["behav_datetime"]}/'
        assert path.exists(rec_dir)

    session_dir = rec_dir + f'{session}.kilosort{metadata["kilosort_ver"]}_probe{metadata["probe_num"]}/'
    preprocess_dir = session_dir + 'preprocessing_output/'
    assert path.exists(preprocess_dir)

    # location of Trodes timestamps (in the kilosort folder of first probe)
    timestamps_dat = rec_dir + f'{session}.kilosort/{session}.timestamps.dat'
    return session_dir, behav_dir, preprocess_dir, timestamps_dat


def get_stitched_session_paths(sesh_1_metadata, sesh_2_metadata):
    session1_dir, behav1_dir, preprocess1_dir, timestamps1_dat = get_session_path(sesh_1_metadata)
    session2_dir, behav2_dir, preprocess2_dir, timestamps2_dat = get_session_path(sesh_2_metadata)

    combined_session = f"{sesh_1_metadata['date']}_{sesh_2_metadata['date']}"
    stitch_dat_dir = session1_dir.split('ephys')[0] + 'ephys/'
    stitch_dir = stitch_dat_dir + f"{combined_session}/probe{sesh_1_metadata['probe_num']}/kilosort{sesh_1_metadata['kilosort_ver']}/"
    assert path.exists(stitch_dir)

    return {'session1_dir':session1_dir, 'behav1_dir':behav1_dir, 'preprocess1_dir':preprocess1_dir, 'timestamps1_dat':timestamps1_dat,
            'session2_dir':session2_dir, 'behav2_dir':behav2_dir, 'preprocess2_dir':preprocess2_dir, 'timestamps2_dat':timestamps2_dat,
            'stitch_dir':stitch_dir}

