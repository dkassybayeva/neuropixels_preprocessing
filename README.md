## Neuropixels Preprocessing
Preprocessing pipeline for Neuropixels recordings using kilosort, additional cluster metrics, Phy2, and export functions to cellbase

### Extracting traces using Trodes

1) Transfer .rec file for days you wish to process from the server. It’s best for this to go to an SSD (i.e., X: or Y: drive).

2) Open Trodes. From the main menu select Open Playback File, and browse to your desired .rec file.

3) File -> extract -> check dio and kilosort -> “start”. This may take a while to show the progress as a percent.  New folders will be created in the same directory as the .rec file, one for DIO and another for kilosort, each containing their respective .dat files.  The kilosort folder also contains the channel map .dat for kilosort as well as the timestamps that will be needed to sync the TTL and behavioral times in the post-spike pipeline.

4) After double checking that the original .rec file is still on the server (with the correct size etc.), delete the local copy to save space. The size of the combined kilosort .dat files is approximately the same as the .rec, so this will save about 0.5TB.

### Spike sorting

1) Run Kilosort 4 either using the GUI directly or using spike_sort_pipeline.py:

   - At the moment, the preferred method is to use the kilosort .dat files exported from Trodes, as described above in Extracting Traces Using Trodes.
     To use the .dat files, set `USE_REC=False` at the top of spike_sort_pipeline.py.  After that, the bare minimum to run
     sorting with Kilosort 4 is to set `RUN_SORTING = True` and then update the paths in the PATHS section,
     including the `if not USE_REC` block, where the probe number can be given.<br/><br/>
   
     Other variables of interest are:
     - `AGGREGATE_SORTING = False`  Only used for `USE_REC=True`.  At the moment should always be set to False because this causes very slow sorting.
     - `FILTER_RAW_BEFORE_SORTING = True`  Should be set to True.  Simply filters the raw data.
     - `SAVE_PREPROCESSING = False`  Can be used as intermediate step to save the preprocessing, but the only processing used at the moment 
     	is the filtering, and that is very fast.  Therefore, setting this  saves disk space and time.
     - `RUN_ANALYSIS = True`  This should be set to True in order to calculate the quality metrics needed for manual curation in Phy.
	   <br/><br/>
   
	- <b>The sorting output should be saved to an SSD drive, otherwise Phy will be very slow.</b>  The default output path is in the main .rec
      folder under sorting_output/.
	  <br/><br/>


2) Manual curation in Phy 
	- Open Anaconda powershell, and change directory to the sorting_output directory 
    - Start the phy anaconda environment (e.g., conda activate phy2, but this may vary on the computer.  Type `$conda env list` to see available environments. 
    - Start phy for clustering: `$ phy template-gui params.py`
      - If an error occurs, make sure that the path at the top of params.py points to the probe's .dat file in the .kilosort folder.
	- Cluster manually 
      - keyboard shortcuts to remember: alt + g labels a spike as good, alt + m labels it as bad, and :merge merges together all the selected clusters. In FeatureView, CTRL+LeftClick to encircle points and then press K to split them. 
      - We prefilter with the following stats: fr > 0.5 & Amplitude > 700 & KSLabel == ‘good’ & ISIv < 3 (see .txt on the server for current filter settings).
	  - When stitching multiple sessions together, the curation can be somewhat lenient: Take a cell if it matches the above filter criteria and
		1) has consistent spike amplitudes,
		2) isn’t obviously two clusters,
		3) has a consistent firing rate across the session break. 
	
      - If it’s a badly aligned day, many cells won't look consistent, especially in the upper part of the probe. 

[//]: # (10&#41; Re-evaluate the cluster metrics by running rerun\_metrics\_after\_cluster\_alteration.m)

### Spike time, TTL, and behavioral timestamp reconciliation 

Scripts for this section are found in the post_spike_sort/ directory.

#### Single session

1) Make sure that the relevant behavior file (BPod session file, e.g., [subject]\_[protocol]\_[monthDay]\_[year]\_Session[#].mat) is in the bpod_session folder (e.g., O:\data\\[subject\]\bpod_session\\[bpod_datetime\]\).
2) Open post\_cluster\_pipeline.py, change the relevant variables and paths up to the PIPELINE heading, and run it.    **--> spike\_mat\_in\_ms.npy**
	- If stitching sessions, set SAVE_INDIVIDUAL_SPIKETRAINS = True

	  **--> spike\_times/spike\_times\_in\_sec\_shank=\[PROBE#\]\_clust=\[UNIT#\].npy**)


#### Stitching sessions
0) Run post\_spike\_sort/extract\_waveforms.m for both individual sessions. (This step is required to compare neurons when validating stitched sessions.) **--> waveforms/unit\_\[UNIT#\].mat**
1) Run misc_utils/combine\_session\_dat_files.m after entering the proper input (to .rec files of the individual sessions) and output paths .  This will create a combined.dat file in the output directory in a subdirectory named \[Session1\]\_\[Session2\].
2) Sort (at the moment, use Kilosort GUI) the combined.dat file and curate in Phy.
3) Run post\_spike\_sort/separate\_session\_spiketimes\_from\_combined\_data.py (requires curated Phy cluster\_group.tsv file in the combined data folder). **--> in combined data folder saves two files: spike_mat_in_ms_\[subject\]_\[session\]_probe\[probe#\]_from_combined_data.npy**
4) Continue with https://github.com/Ott-Decision-Circuits-Lab/spike_response_analysis/tree/master/session_stitching
