"""
Extract electrode locations from probe mapping output of HERBS

Created on Mon Jun  5 17:10:29 2023

@author: Greg Knoll
"""

import pandas as pd

rat = 'R31'
PROBE_PKL_PATH = r'C:/Users/science person/Documents/HERBS/slices/' + f'{rat}/'

ROI = 'Prelimbic area'

probe_pkl = pd.read_pickle(PROBE_PKL_PATH + 'probe.pkl')

probe_data = probe_pkl['data']

ROI_idx = [i for i,name in enumerate(probe_data['label_name']) if name==ROI][0]

ROI_start = sum(probe_data['region_sites'][:ROI_idx])
ROI_end = sum(probe_data['region_sites'][:ROI_idx+1])

probe_df = pd.DataFrame(columns=['Region', 'Lowest Electrode', 'Highest Electrode'])
probe_df.loc[0] = [ROI, int(ROI_start), int(ROI_end)]
filename = 'probe_electrodes_in' + ROI.replace(' ', '_').lower() + '.csv'
probe_df.to_csv(PROBE_PKL_PATH + filename, index=False, header=True)

# @TODO: Save all regions
# @TODO: load the good Phy units and label them with regions (save in spike_mat or somewhere in dataobject)