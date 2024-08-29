from os import getlogin
import pandas as pd

rat = 'Nina2'
session = '20210625_114657'
probe = 1

base_dir = f'/home/{getlogin()}/Workspace/ott_neuropix_data/data_cache/'
probe_dir = base_dir + f'{rat}/ephys/{session}.rec/preprocessing_output/probe{probe}/'

WU_df = pd.read_csv(base_dir + 'electrode_coord_WUSTLdataset.csv')
phy_df = pd.read_csv(probe_dir + 'cluster_info.tsv', sep='\t')

good_units = phy_df[phy_df['group']=='good']
good_units.reset_index(drop=True, inplace=True)
good_units = good_units.assign(region="", AP="", ML="", DV="")

rat_electrode_df = WU_df[(WU_df.Rat==rat) & (WU_df.Probe==probe)]

for i, row in good_units.iterrows():
    good_units.loc[i, 'region'] = rat_electrode_df.loc[rat_electrode_df.Electrode == row['ch'], 'Region'].values[0]
    good_units.loc[i, 'AP']     = rat_electrode_df.loc[rat_electrode_df.Electrode == row['ch'], 'AP'].values[0]
    good_units.loc[i, 'ML']     = rat_electrode_df.loc[rat_electrode_df.Electrode == row['ch'], 'ML'].values[0]
    good_units.loc[i, 'DV']     = rat_electrode_df.loc[rat_electrode_df.Electrode == row['ch'], 'DV'].values[0]

good_units.to_csv(probe_dir + f'{rat}_{session}_probe{probe}--good_unit_stats_w_region.tsv', sep='\t', index=False)