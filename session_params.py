from os import path, makedirs, getlogin
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
trace_subsample_bin_size_ms = 25 # sample period in ms
sps = 1000 / trace_subsample_bin_size_ms  # (samples per second) resolution of aligned traces

# -------params for trace interpolation------- #
"""
For the source of these numbers, see 'Temporal dynaics clustering' in Hirokawa et al. Nature (2019) in Methods.
"""
interpolation_param_dict = dict(
    trial_times_in_reference_to='TrialStart',  # ['TrialStart', 'ResponseStart']
    resp_start_align_buffer=None,  # for ResponseStart
    trial_event_interpolation_lengths = [
        int(0.5 * sps),  # ITI->center poke
        int(0.45 * sps), # center->stim_begin
        int(.5 * sps),   # stim delivery
        int(.3 * sps),   # movement to side port
        # int(0.5 * sps),  # first 0.5s of anticipation epoch
        # int(0.5 * sps),  # second part of anticipation epoch warped into 0.5s (actually half second in reward-bias)
        int(3.0 * sps),  # anticipation epoch
        int(1.5 * sps),  # after feedback
    ],
    pre_center_interval = int(0.5 * sps),
    post_response_interval = None,  # int(0.5 * sps) or None.  If None, then the midpoint between response start and end is used
    downsample_dt=trace_subsample_bin_size_ms,
)

alignment_param_dict = dict(
    trial_times_in_reference_to='TrialStart',  # ['TrialStart', 'ResponseStart']
    resp_start_align_buffer=None,  # for ResponseStart
    downsample_dt=trace_subsample_bin_size_ms,
    pre_stim_interval = int(0.5 * sps),  # truncated at center_poke
    post_stim_interval = int(0.5*sps),  # truncated at stim_off
    pre_response_interval = int(3.0*sps),  # truncated at stim_off
    post_response_interval = int(4.0*sps),  # truncated at response_end
    pre_reward_interval = int(6.0*sps),  # truncated at response_time
    post_reward_interval = int(5.0*sps),  # truncated at trial_end
)



# ---------file names---------- #
spike_mat_str_indiv = f'spike_mat_in_ms.npy'
gap_filename = f"trodes_intersample_periods_longer_than_{max_ISI}s.npy"
# ----------------------------------------------------------------------- #

def get_root_path(data_root):
    if data_root=='server':
        data_root='O:data/'
        if not path.exists(data_root):
            data_root='/media/ottlab/data/'
    elif data_root=='local':
        print('USING LOCAL DATA FOR TESTING!!!')
        data_root = f'/home/{getlogin()}/Workspace/ott_neuropix_data/'
    else:
        data_root = data_root + 'Neurodata/'
    return data_root


def save_directory_helper(data_root):
    if data_root=='server':
        data_root = 'O:share/ephys/'
        if not path.exists(data_root):
            data_root = '/media/ottlab/share/ephys/'
    elif data_root == 'local':
        data_root = f'/home/{getlogin()}/Workspace/ott_neuropix_data/'
    return data_root


def write_session_metadata_to_csv(data_root):
    columns = ['ott_lab', 'rat_name', 'date', 'experimenter', 'region',
               'trodes_datetime', 'trodes_logfile', 'trodes_config', 'recording_type',
               'n_probes', 'DIO_port_num', 'kilosort_ver',
               'behav_datetime', 'task', 'behavior_mat_file']

    metadata = dict(
        ott_lab = False,
        rat_name = 'Nina2',
        date = '20210623',
        experimenter = 'Amy',
        region = 'lOFC',
        # ----------------------------------- #
        trodes_datetime = '20210623_121426',
        trodes_logfile = '',
        trodes_config = '',
        recording_type = 'neuropixels_1.0',
        n_probes = 2,
        DIO_port_num = 6,
        kilosort_ver = 2.5,
        # ----------------------------------- #
        behav_datetime = '20210623',
        task = 'time-investment', # ['matching', 'reward-bias', 'time-investment']
        # ----------------------------------- #
    )

    task_type = 'TwoArmBanditVariant' if metadata['task'] == 'matching' else 'DiscriminationConfidence'
    metadata['behavior_mat_file'] = f'{metadata["rat_name"]}_{task_type}_{metadata["behav_datetime"]}.mat'

    DATA_DIR = save_directory_helper(data_root)

    ephys_metadata_file = DATA_DIR + 'ephys_sessions_metadata.csv'
    try:
        ephys_df = pd.read_csv(ephys_metadata_file)
    except:
        ephys_df = pd.DataFrame(columns=columns)

    ephys_df.loc[len(ephys_df)] = metadata
    ephys_df.to_csv(ephys_metadata_file, index=False)

    return metadata


def load_session_metadata_from_csv(data_root, rat, session_date):
    DATA_DIR = save_directory_helper(data_root)
    ephys_metadata_file = DATA_DIR + 'ephys_sessions_metadata.csv'
    ephys_df = pd.read_csv(ephys_metadata_file)

    rat_idx = ephys_df['rat_name'].apply(str) == rat
    date_idx = ephys_df['date'].apply(str) == session_date
    return ephys_df[rat_idx & date_idx].iloc[0].to_dict()


def insert_value_into_metadata_csv(data_root, rat, session_date, column, value):
    DATA_DIR = save_directory_helper(data_root)
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
        rec_dir = root_path + f'{rat}/ephys/{e_session}.rec/'
        session_paths = dict(
            rec_dir = rec_dir,
            probe_dir = rec_dir + f'{e_session}.kilosort{metadata["kilosort_ver"]}_probe{metadata["probe_num"]}/',
            preprocess_dir = rec_dir + 'preprocessing_output/',
            timestamps_dat = rec_dir + f'{e_session}.kilosort/{e_session}.timestamps.dat',  # Trodes timestamps in general KS dir
        )
        assert path.exists(session_paths['rec_dir'])
    else:
        session_paths = dict()

    session_paths['behav_dir'] = root_path + f'{rat}/bpod_session/{metadata["behav_datetime"]}/'
    assert path.exists(session_paths['behav_dir'])

    return session_paths


def get_stitched_session_paths(data_root, sesh_1_metadata, sesh_2_metadata):
    session1_paths = get_session_path(sesh_1_metadata, data_root, is_ephys_session=True)
    session2_paths = get_session_path(sesh_2_metadata, data_root, is_ephys_session=True)
    rat = sesh_1_metadata['rat_name']

    stitch_paths = dict()
    combined_session = f"{sesh_1_metadata['date']}_{sesh_2_metadata['date']}"
    root_path = save_directory_helper(data_root)
    stitch_paths['stitch_dir'] = root_path + f'{rat}/{combined_session}/'
    assert path.exists(stitch_paths['stitch_dir'])

    stitch_paths['probe_dir'] = stitch_paths['stitch_dir'] + f"probe{sesh_1_metadata['probe_num']}/kilosort{sesh_1_metadata['kilosort_ver']}/"
    stitch_paths['preprocess_dir'] = stitch_paths['stitch_dir'] + f"preprocessing_output/probe{sesh_1_metadata['probe_num']}/"
    stitch_paths['matches_dir'] = stitch_paths['preprocess_dir'] + 'matches/'

    return {f"{sesh_1_metadata['date']}":session1_paths, f"{sesh_2_metadata['date']}":session2_paths, 'stitched':stitch_paths}

