# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:33:50 2022

@author: Greg Knoll
"""

import neuropixels_preprocessing.lib.timing_utils as timing_utils
import neuropixels_preprocessing.lib.obj_utils as obj_utils

fs = 30000.  # sampling frequency of Trodes

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

cellbase_dir = session_path + 'cellbase/'
obj_utils.make_dir_if_nonexistent(cellbase_dir)
#----------------------------------------------------------------------#

timing_utils.create_spike_mat(session_path, kilosort_dir, date, probe_num, fs)