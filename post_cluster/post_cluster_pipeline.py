# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""

import neuropixels_preprocessing.lib.timing_utils as tu
import neuropixels_preprocessing.lib.obj_utils as ou

fs = 30000.  # sampling frequency of Trodes
max_ISI = 0.001  # max intersample interval (ISI), above which the period
                 # was considered a "gap" in the recording

#----------------------------------------------------------------------#
#                           PATHS
#----------------------------------------------------------------------#
DATAPATH = 'X:/Neurodata'
rat_name = 'Nina2'
date = '20210623_121426'
probe_num = 2

rec_file_path = f"{DATAPATH}/{rat_name}/{date}.rec/"
kilosort_dir = rec_file_path + f"{date}" + ".kilosort_probe{}/"
session_path = kilosort_dir.format(probe_num)

# location of Trodes timestamps (in the kilosort folder of first probe)
timestamp_file = kilosort_dir.format(1) + date + '.timestamps.dat'

cellbase_dir = session_path + 'cellbase/'
ou.make_dir_if_nonexistent(cellbase_dir)
#----------------------------------------------------------------------#

tu.create_spike_mat(session_path, timestamp_file, date, probe_num, fs, 
                    save_individual_spiketrains=False)

gap_filename = tu.find_recording_gaps(timestamp_file, fs, max_ISI, cellbase_dir)

tu.extract_TTL_events(session_path, gap_filename, save_dir=cellbase_dir)