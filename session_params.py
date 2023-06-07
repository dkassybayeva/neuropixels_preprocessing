from os import path, makedirs


# ----------------------------------------------------------------------- #
#                           Constants
# ----------------------------------------------------------------------- #
fs = 30e3  # Trodes sampling frequency in Hz
ms_converter = 1000 / fs
n_Npix1_electrodes = 960
n_active_electrodes = 384
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
#                           Session Info
# ----------------------------------------------------------------------- #
OTT_LAB_DATA = True
rat = '1'
probe = '1'
kilosort_ver = '2.5'
session1 = '20230506_152707'
# session1 = '20230507_123146'

Trodes_config = f'D:NeuroData/R{rat}/2023-04-10.trodesconf'

# name of the BPod behavioral data file for session1
behav_datetime = '20230506_151733'
task = 'matching'
behavior_mat_file = f'1_TwoArmBanditVariant_{behav_datetime}.mat'
# behav_datetime = '20230507_122316'
# task = 'time-investment'
# behavior_mat_file = f'1_DiscriminationConfidence_{behav_datetime}.mat'

# when stitching
STITCH_SESSIONS = False
if STITCH_SESSIONS:
    session2 = '20230507_123146'
    combined_session = '20230506_20230507'
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#               Create metadata to save with data object               #
#----------------------------------------------------------------------#
recording_session_id = 0

if OTT_LAB_DATA:
    metadata = {'ott_lab': True,
                'task': task,
                'experimenter': 'Gregory Knoll',
                'region': 'mPFC',
                'recording_type': 'neuropixels_1.0',
                'rat_name': f'R{rat}',
                'date': session1,
                'probe_num': probe,
                'DIO_port_num': 1,
                }
else:
    metadata = {'ott_lab': False,
                'time_investment': False,
                'reward_bias': False,
                'task': task,
                'prior': False,  # Could possibly be Amy's code for a task type that was previously used
                'experimenter': 'Amy',
                'region': 'lOFC',
                'recording_type': 'neuropixels',
                'experiment_id': 'learning_uncertainty',
                'linking_group': 'Nina2',
                'rat_name': rat_name,
                'date': date,
                'probe_num': probe_num,
                }

    #-----------------------------------#
    # List of stimuli for each experiment:
    #     'freq' =
    #     'freq_nat' =
    #     'nat' =
    #     'nat_nat' =
    #-----------------------------------#
    metadata['stimulus'] = 'freq'

    #-----------------------------------#
    # Session number code:
    #     -1 = good performance, no noise (or rare)
    #      0 = first day with noise
    #     -2 = poor performance, no noise (<70%)
    #-----------------------------------#
    metadata['behavior_phase'] = -1
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
#                               Paths
# ----------------------------------------------------------------------- #
try:
    INDIV_DATA_DIR = f'D:NeuroData/R{rat}/'
    assert path.exists(INDIV_DATA_DIR)
    REC_PATH = BEHAV_PATH = INDIV_DATA_DIR + f'{session1}.rec/'
except:
    INDIV_DATA_DIR = f'/media/ottlab/data/{rat}/'
    assert path.exists(INDIV_DATA_DIR)
    REC_PATH = INDIV_DATA_DIR + f'ephys/{session1}.rec/'
    BEHAV_PATH = INDIV_DATA_DIR + f'bpod_session/{behav_datetime}/'

SESSION_DIR = REC_PATH + f'{session1}.kilosort{kilosort_ver}_probe{probe}/'
PREPROCESS_DIR = SESSION_DIR + 'preprocessing_output/'
assert path.exists(PREPROCESS_DIR)

# location of Trodes timestamps (in the kilosort folder of first probe)
timestamps_dat = REC_PATH + f'{session1}.kilosort/{session1}.timestamps.dat'

# -----------------When stitching sessions------------------ #
if STITCH_SESSIONS:
    try:
        INDIV_DATA_DIR2 = f'Y:NeuroData/R{rat}/'
        assert path.exists(INDIV_DATA_DIR2)
        STITCH_DAT_DIR = f'D:NeuroData/R{rat}/'
        assert path.exists(STITCH_DAT_DIR)
    except:
        INDIV_DATA_DIR2 = STITCH_DAT_DIR = INDIV_DATA_DIR

    REC_PATH2 = INDIV_DATA_DIR2 + f'{session2}.rec/'
    SESSION_DIR2 = REC_PATH2 + f'{session2}.kilosort{kilosort_ver}_probe{probe}/'
    PREPROCESS_DIR2 = SESSION_DIR2 + 'preprocessing_output/'

    STITCH_DIR = STITCH_DAT_DIR + f'{combined_session}/probe{probe}/kilosort{kilosort_ver}/'
# ----------------------------------------------------------------------- #
