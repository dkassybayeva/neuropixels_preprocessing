from os import path, makedirs


# ----------------------------------------------------------------------- #
#                           Constants
# ----------------------------------------------------------------------- #
fs = 30e3  # Trodes sampling frequency in Hz
ms_converter = 1000 / fs
# ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------- #
#                           Session Info
# ----------------------------------------------------------------------- #
OTT_LAB_DATA = True
rat = '1'
probe = '1'
kilosort_ver = '2.5'
session1 = '20230506_152707'
# name of the BPod behavioral data file for session1
behavior_mat_file = '1_TwoArmBanditVariant_20230506_151733.mat'

# when stitching
STITCH_SESSIONS = True
if STITCH_SESSIONS:
    session2 = '20230507_123146'
    combined_session = '20230506_20230507'
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#               Create metadata to save with data object               #
#----------------------------------------------------------------------#
recording_session_id = 0

if OTT_LAB_DATA:
    metadata = {'task': 'matching',
                'experimenter': 'Gregory Knoll',
                'region': 'mPFC',
                'recording_type': 'neuropixels_1.0',
                'rat_name': f'R{rat}',
                'date': session1,
                'probe_num': probe,
                }
else:
    metadata = {'time_investment': False,
                'reward_bias': False,
                'prior': False,  # Could possibly be Amy's code for a task type that was previously used
                'experimenter': 'Amy',
                'region': 'lOFC',
                'recording_type': 'neuropixels',
                'experiment_id': 'learning_uncertainty',
                'linking_group': 'Nina2',
                'rat_name': rat_name,
                'date': date,
                'sps': sps,
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
except:
    INDIV_DATA_DIR = f'/media/ottlab/data/{rat}/ephys/'
    assert path.exists(INDIV_DATA_DIR)

REC_PATH = INDIV_DATA_DIR + f'{session1}.rec/{session1}'
SESSION_DIR = REC_PATH + f'.kilosort{kilosort_ver}_probe{probe}/'
PREPROCESS_DIR = SESSION_DIR + 'preprocessing_output/'
assert path.exists(PREPROCESS_DIR)

# location of Trodes timestamps (in the kilosort folder of first probe)
timestamps_dat = REC_PATH + f'.kilosort/{session1}.timestamps.dat'

# -----------------When stitching sessions------------------ #
if STITCH_SESSIONS:
    try:
        INDIV_DATA_DIR2 = f'Y:NeuroData/R{rat}/'
        assert path.exists(INDIV_DATA_DIR2)
        STITCH_DAT_DIR = f'D:NeuroData/R{rat}/'
        assert path.exists(STITCH_DAT_DIR)
    except:
        INDIV_DATA_DIR2 = STITCH_DAT_DIR = INDIV_DATA_DIR

    REC_PATH2 = INDIV_DATA_DIR2 + f'{session2}.rec/{session2}'
    SESSION_DIR2 = REC_PATH2 + f'.kilosort{kilosort_ver}_probe{probe}/'
    PREPROCESS_DIR2 = SESSION_DIR2 + 'preprocessing_output/'

    STITCH_DIR = STITCH_DAT_DIR + f'{combined_session}/probe{probe}/kilosort{kilosort_ver}/'
# ----------------------------------------------------------------------- #
