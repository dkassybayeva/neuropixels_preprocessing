"""
Extract electrode locations from probe mapping output of HERBS

Created on Mon Jun  5 17:10:29 2023

@author: Greg Knoll
"""

import pandas as pd

from neuropixels_preprocessing.session_params import *

rat = 'R1'
PROBE_PKL_PATH = r'C:/Users/science person/Documents/HERBS/slices/' + f'{rat}/'

probe_pkl = pd.read_pickle(PROBE_PKL_PATH + 'probe.pkl')
probe_data = probe_pkl['data']
probe_df = pd.DataFrame(columns=['Region', 'Lowest Electrode', 'Highest Electrode'])


for ROI_idx, ROI in enumerate(probe_data['label_name']):
    ROI_start = sum(probe_data['region_sites'][:ROI_idx])
    ROI_end = sum(probe_data['region_sites'][:ROI_idx+1])
    new_row = pd.DataFrame([[ROI, int(ROI_start), int(ROI_end)]], columns=probe_df.columns)
    
    probe_df = pd.concat([probe_df, new_row], ignore_index=True)
    
filename = f'{rat}_probe_electrode_locations.csv'
probe_df.to_csv(PROBE_PKL_PATH + filename, index=False, header=True)


# @TODO: load the good Phy units and label them with regions (save in spike_mat or somewhere in dataobject)