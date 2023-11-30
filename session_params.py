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

# ---------file names---------- #
spike_mat_str_indiv = f'spike_mat_in_ms.npy'
gap_filename = f"trodes_intersample_periods_longer_than_{max_ISI}s.npy"
# ----------------------------------------------------------------------- #

def get_root_path(data_root):
    if data_root=='server':
        data_root='O:data/'
        if ~path.exists(data_root):
            data_root='/media/ottlab/data/'
    elif data_root=='local':
        print('USING LOCAL DATA FOR TESTING!!!')
        data_root = '/home/mud/Workspace/ott_neuropix_data/'
    else:
        data_root = data_root + 'Neurodata/'
    return data_root


def save_directory_helper():
    DATA_DIR = f'/media/ottlab/share/ephys/'
    if ~path.exists(DATA_DIR):
        DATA_DIR = 'O:share/ephys/'
    return DATA_DIR


def write_session_metadata_to_csv():
    columns = ['ott_lab', 'rat_name', 'date', 'trodes_datetime', 
               'trodes_logfile', 'task', 'behav_datetime',
               'region', 'n_probes', 'trodes_config', 'recording_type', 'DIO_port_num',
               'time_investment', 'reward_bias', 'behavior_mat_file',
               'sps', 'kilosort_ver', 'experimenter',
               'linking_group', 'recording_session_id', 'prior', 'experiment_id', 'stimulus', 'behavior_phase']

    metadata = {'ott_lab': False,
                'rat_name': 'Nina2',
                'date': '20210625',
                # ----------------------------------- #
                'trodes_datetime': '20210625_114657',
                'trodes_logfile': 'Trodes_extraction_log.txt',
                'trodes_config': '',
                'recording_type': 'neuropixels_1.0',
                'n_probes': 2,
                'DIO_port_num': 6,
                # ----------------------------------- #
                'behav_datetime': '20210625',
                'task': 'reward-bias', # ['matching', 'reward-bias', 'time-investment']
                'time_investment': False,
                'reward_bias': True,
                # ----------------------------------- #
                'prior': False,  # Could possibly be Amy's code for a task type that was previously used
                'experiment_id': 'learning_uncertainty',
                'linking_group': None, #'Nina2',
                'recording_session_id': 0,
                'kilosort_ver': 2.5,
                'sps': sps
    }

    task_type = 'TwoArmBanditVariant' if metadata['task'] == 'matching' else 'DiscriminationConfidence'
    metadata['behavior_mat_file'] = f'{metadata["rat_name"]}_{task_type}_{metadata["behav_datetime"]}.mat'
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


def insert_value_into_metadata_csv(rat, session_date, column, value):
    DATA_DIR = save_directory_helper()
    ephys_metadata_file = DATA_DIR + 'ephys_sessions_metadata.csv'
    ephys_df = pd.read_csv(ephys_metadata_file)

    session_idx = (ephys_df['rat_name'].apply(str)==rat) & (ephys_df['date'].apply(str)==str(session_date))
    ephys_df.loc[session_idx, column] = value
    ephys_df.to_csv(ephys_metadata_file, index=False)


def get_session_path(metadata, data_root, is_ephys_session):
    rat = metadata['rat_name']
    if data_root=='server' and 'R' in rat:
        rat = rat.split('R')[-1]

    root_path = get_root_path(data_root)

    if is_ephys_session:
        e_session = metadata['trodes_datetime']
        rec_dir = root_path + f'{rat}/ephys/{e_session}.rec/',
        session_paths = dict(
            rec_dir = rec_dir,
            probe_dir = rec_dir + f'{e_session}.kilosort{metadata["kilosort_ver"]}_probe{metadata["probe_num"]}/',
            preprocess_dir = rec_dir + 'preprocessing_output/',
            timestamps_dat = rec_dir + f'{e_session}.kilosort/{e_session}.timestamps.dat',  # Trodes timestamps in general KS dir
        )
    else:
        session_paths = dict()

    session_paths['behav_dir'] = root_path + f'{rat}/bpod_session/{metadata["behav_datetime"]}/'

    for path_i in session_paths.values():
        assert path.exists(path_i)
    return session_paths


def get_stitched_session_paths(sesh_1_metadata, sesh_2_metadata):
    session1_paths = get_session_path(sesh_1_metadata)
    session2_paths = get_session_path(sesh_2_metadata)
    rat = sesh_1_metadata['rat_name']

    stitch_paths = dict()
    combined_session = f"{sesh_1_metadata['date']}_{sesh_2_metadata['date']}"
    try:
        stitch_paths['stitch_dir'] = f'O:data/{rat}/ephys/{combined_session}/'
        assert path.exists(stitch_paths['stitch_dir'])
    except:
        if 'R' in rat:
            rat = rat.split('R')[-1]
        stitch_paths['stitch_dir'] = f'/media/ottlab/data/{rat}/ephys/{combined_session}/'
        assert path.exists(stitch_paths['stitch_dir'])

    stitch_paths['probe_dir'] = stitch_paths['stitch_dir'] + f"probe{sesh_1_metadata['probe_num']}/kilosort{sesh_1_metadata['kilosort_ver']}/"
    stitch_paths['preprocess_dir'] = stitch_paths['stitch_dir'] + f"preprocessing_output/probe{sesh_1_metadata['probe_num']}/"
    stitch_paths['matches_dir'] = stitch_paths['preprocess_dir'] + 'matches/'

    return {f"{sesh_1_metadata['date']}":session1_paths, f"{sesh_2_metadata['date']}":session2_paths, 'stitched':stitch_paths}

