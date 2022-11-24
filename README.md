# neuropixels_preprocessing
Preprocessing pipeline for Neuropixels recordings using kilosort, additional cluster metrics, Phy2, and export functions to cellbase

## Extracting Traces Using Trodes

1) Transfer .rec file for days you wish to process from the server. It’s best for this to go to an SSD (i.e., X: or Y: drive).

2) Open Trodes. From the main menu select Open Playback File, and browse to your desired .rec file.

3) File -> extract -> analgio, dio, kilosort, “start”. This takes a few hours to run on an SSD and overnight on an HDD.  This will result in new folders in the same directory as the .rec file, one for DIO, analog, and kilosort, each containing their respective .dat files.

4) After double checking the backup of the .rec file is still on the server (and correct size etc.), delete the local copy to save space. 

## Clustering:

5) Run either main\_kilosort\_25\_Torben.m or main\_kilosort\_25\_Batch.m, depending on whether you are processing one day, or processing multiple days. You must edit the file paths in the script. It’s good for the temporary files to be located on an SSD for speed, but the KS output file doesn’t have to be. 

6) Open anaconda powershell, and change directory to the Kilosort (KS) output directory

7) Start the phy anaconda environment (conda activate phy2)

8) Start phy for clustering (phy template-gui params.py)

9) Cluster manually

	a) Keyboard shortcuts to remember: alt + g labels a spike as good, alt + m labels it as bad, and :merge merges together all the selected clusters. 
	
	b) We prefilter with the following stats: fr > 0.5 & Amplitude > 700 & KSLabel == ‘good’ & ISIv < 3
	
	c) For aligning multiple days I’m somewhat lenient. I basically take a cell if it matches those criteria and:
	
	- has consistent spike amplitudes
	
	- isn’t obviously two clusters
	
	- and has consistent firing rate across the session break. 
	
	If it’s a badly aligned day, many cells won't look consistent, especially in the upper part of the probe. 


## After Clustering:

- Copy convert_spikes_pkl_to_mat_file.py from this repository to the Kilosort output directory (e.g., X:\NeuroData\SubjectName\date_time.rec\data_time.kilosort_probe1\)
and run it (e.g., cmd: python convert_spikes.py) -> spikes_per_cluster.mat

- Run MakeTTNeuropixel(\_batchalign).m in Matlab, editing directories as relevant.  This creates spike time vectors in the cellbase subdirectory, matching the Kilosort/Phy cluster information with the timekeeping from Trodes.  This data is saved for each unit in the cellbase directory under TT[shank#]\_[clusterID].mat.  It also saves the waveforms in WF[shank#]\_[clusterID].mat, as well as "gaps" (GAPS.mat), the cluster quality metrics (PhyLabels\_[shank#].mat) and the analog input TTL events (EVENTS.mat).

- For each day (from cellbase directory):
	
	a) Copy relevant behavior file (BPod session file) to the cellbase directory (e.g., [subject]\_[protocol]\_[monthDay]\_[year]\_Session[#].mat)
 	
	b) Run MakeTrialEventsNeuropixels.m on cellbase directory -> creates TE.mat, TEbis.mat, and TrialEvents.mat, as well as two Aligned*.mat files.  TrialEvents.mat has the extracted trial events data.
	
	c) If 2nd day in alignment MakeTrialEvents2TorbenNP needs to be edited to say: Events\_TTL2 Events\_TS2 on line 45, Events\_TTL1 Events\_TS1 if first day
	
- Run Amy's scripts:
	- MATLAB/processTrialEventsDual2AFC.m **--> RecBehav.mat**
	    - curates results in to TrialEvents.mat of all trials from Bpod
	    - adds some [1 x nTrials] fields 
	    - removes any Bpod settings structs or anything that isn't [1 x nTrials] array
	
	- MATLAB/process\_TTcellbase.m **--> traces\_ms.mat**
	
	    - combines all TT[shank#]\_[clusterID].mat files created by MakeTTNeuropixel(\_batchalign).m into a signle binary matrix with spike times
	
	- PYTHON/process\_acdat.py: uses **traces\_ms.m** and **RecBehav.mat**
	
	    - requires (pip install): mat73, imblearn
	    - aligns the spiking and behavioral data to different events (e.g., trial start vs response start)
	    - collects all data (including spiking and behavioral data [now pandas DataFrame]) into a DataContainer object specific to the experiment type (e.g., 2AFC) and saves the entire object using pickle
