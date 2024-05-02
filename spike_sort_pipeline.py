import probeinterface
import spikeinterface.full as si
from spikeinterface.extractors import read_spikegadgets
from spikeinterface.extractors import SpikeGadgetsRecordingExtractor
import spikeinterface.widgets as sw
from probeinterface import get_probe
from probeinterface.plotting import plot_probe, plot_probe_group
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from xml.etree import ElementTree
import pandas as pd
import platform


# -------------------------------------------------------------------------- #
#                         Script Parameters
# -------------------------------------------------------------------------- #
USE_REC = True
FILTER_RAW_BEFORE_SORTING = True  # applies HPF and CMR
SAVE_PREPROCESSING = False

RUN_SORTING = True
AGGREGATE_SORTING = False
RUN_ANALYSIS = True
EXPORT_TO_PHY = False

FILTER_GOOD_UNITS = False
ONLINE_CURATION = False

PLOT_PROBE = False
PLOT_BIG_HEATMAPS = False
PLOT_SOME_CHANNELS = False
PLOT_NOISE = False
PLOT_PEAKS_ON_ELECTRODES = False


job_kwargs = dict(chunk_duration='1s', progress_bar=True)
if platform.system() != 'Windows':
    job_kwargs['n_jobs'] = 40   
    
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
#                               Paths
# -------------------------------------------------------------------------- #
base_folder = Path('Y:/NeuroData/TQ03/20210616_115352.rec')
rec_file = base_folder / '20210616_115352.rec'
sorting_folder = base_folder / 'spike_interface_output'

if not USE_REC:
    probe_num = 1
    binary_file = base_folder / f'TQ03_20210617_combined.probe{probe_num}.dat'
    chan_map_file = base_folder / f'TQ03_20210617_combined.channelmap_probe{probe_num}.dat'
# -------------------------------------------------------------------------- #
    
print('Sorting', rec_file)


if USE_REC:
    # stream_names, stream_ids = si.get_neo_streams('spikegadgets', rec_file)
    # print('Available streams', stream_ids)
    
    
    raw_dat = read_spikegadgets(rec_file)
    
    # tvec = raw_dat.get_times()
    # channels = raw_dat.get_channel_ids()
    # duration_s = raw_dat.get_total_duration()
    # raw_dat.get_probe().to_dataframe()
    # fs = raw_dat.get_sampling_frequency()

    if PLOT_PROBE:
        plot_probe_group(raw_dat.get_probegroup(), same_axes=True)
        plt.show()
else:
    from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
    from probeinterface import Probe
    from probeinterface.io import parse_spikegadgets_header
    from xml.etree import ElementTree
    
    CONTACT_WIDTH = 16  # um
    CONTACT_HEIGHT = 20  # um
    
    # ------------------------------------ #
    #           Read in data
    # ------------------------------------ #
    spikegadgets_header = parse_spikegadgets_header(rec_file)

    
    root = ElementTree.fromstring(spikegadgets_header)
    sconf = root.find("SpikeConfiguration")
    scaling_from_binary_to_uV = float(sconf[0].attrib['spikeScalingToUv'])
    
    raw_dat = si.read_binary(file_paths=binary_file,
                             sampling_frequency=30_000., 
                             num_channels=384, 
                             dtype='int16',
                             gain_to_uV=scaling_from_binary_to_uV,
                             offset_to_uV=0)
    
    # ------------------------------------ #
    #           Construct Probe
    # ------------------------------------ #
    pad_coords_in_um = readTrodesExtractedDataFile(chan_map_file)['data']
    pad_coords_in_um = np.vstack([np.array(list(row)) for row in pad_coords_in_um])
    n_chan = pad_coords_in_um.shape[0]

    # Construct Probe object
    probe = Probe(ndim=2, si_units="um", model_name="Neuropixels 1.0", manufacturer="IMEC")
    probe.set_contacts(
        contact_ids=np.arange(n_chan),
        positions=pad_coords_in_um[:, :2],
        shapes="square",
        shank_ids=None,
        shape_params={"width": CONTACT_WIDTH, "height": CONTACT_HEIGHT},
    )

    # Wire it (i.e., point contact/electrode ids to corresponding hardware/channel ids)
    probe.set_device_channel_indices(np.arange(n_chan))

    # Create a nice polygon background when plotting the probes
    x_min = probe.contact_positions[:, 0].min()
    x_max = probe.contact_positions[:, 0].max()
    x_mid = 0.5 * (x_max + x_min)
    y_min = probe.contact_positions[:, 1].min()
    y_max = probe.contact_positions[:, 1].max()
    polygon_default = [
        (x_min - CONTACT_WIDTH, y_min - CONTACT_HEIGHT / 2),
        (x_mid, y_min - 5 * CONTACT_HEIGHT),
        (x_max + CONTACT_WIDTH, y_min - CONTACT_HEIGHT / 2),
        (x_max + CONTACT_WIDTH, y_max + CONTACT_WIDTH),
        (x_min - CONTACT_WIDTH, y_max + CONTACT_WIDTH),
    ]
    probe.set_planar_contour(polygon_default)

    # ------------------------------------ #
    #        Attach Probe to data
    # ------------------------------------ #
    raw_dat.set_probe(probe, in_place=True)

    if PLOT_PROBE:
        plot_probe(probe)
        plt.show()


if RUN_SORTING and FILTER_RAW_BEFORE_SORTING:
    try:
        rec_phaseshift = si.phase_shift(raw_dat)
    except:
        rec_phaseshift = raw_dat
    rec_hpf = si.highpass_filter(rec_phaseshift, freq_min=400.)
    
    
    # bad_channel_ids, channel_labels = si.detect_bad_channels(rec_hpf)
    # rec_hpf_good = rec_hpf.remove_channels(bad_channel_ids)
    # print('bad_channel_ids', bad_channel_ids)
    
    recording = si.common_reference(rec_hpf, operator="median", reference="global")
else:
    recording = raw_dat

# here we use static plot using matplotlib backend
if PLOT_BIG_HEATMAPS:
    if FILTER_RAW_BEFORE_SORTING:
        fig, axs = plt.subplots(ncols=3, figsize=(20, 10))
        
        si.plot_traces(raw_dat, backend='matplotlib', time_range=(4,5),  clim=(-4000, 4000), ax=axs[0])
        si.plot_traces(rec_hpf, backend='matplotlib', time_range=(4,5),  clim=(-4000, 4000), ax=axs[1])
        si.plot_traces(recording, backend='matplotlib', time_range=(4,5), clim=(-4000, 4000), ax=axs[2])
        for i, label in enumerate(('raw', 'filter', 'cmr')):
            axs[i].set_title(label)
    else:
        si.plot_traces(raw_dat, backend='matplotlib', time_range=(4,6),  clim=(-4000, 4000))
    plt.show()


if PLOT_SOME_CHANNELS:
    some_chans = raw_dat.channel_ids[[100, 150, 200, ]]
    # si.plot_traces({'filter':rec_hpf, 'cmr': rec_cmr}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans, color='k')
    # si.plot_traces({'cmr': rec_cmr}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans, color='k')
    w_ts = sw.plot_timeseries(recording, time_range=(0, 5), channel_ids=some_chans, color='k')
    plt.show()


if SAVE_PREPROCESSING:
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    recording = recording.save(folder=base_folder / 'preprocess', format='binary', **job_kwargs)

    # our recording now points to the new binary folder (if it was saved)
    print(recording)


# we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
if PLOT_NOISE:
    fig, ax = plt.subplots()
    if USE_REC:
        noise_levels_uV = si.get_noise_levels(recording, return_scaled=True)
        _ = ax.hist(noise_levels_uV)  #, bins=np.arange(5, 30, 2.5))
        ax.set_xlabel('noise  [microV]')
    else:
        noise_levels_int16 = si.get_noise_levels(recording, return_scaled=False)
        _ = ax.hist(noise_levels_int16)
        ax.set_xlabel('noise  [a.u.]')
    plt.show()


if PLOT_PEAKS_ON_ELECTRODES:
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    peaks = detect_peaks(rec_hpf,  method='locally_exclusive', noise_levels=noise_levels_int16, detect_threshold=5, radius_um=50., **job_kwargs)
    print(peaks)

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    peak_locations = localize_peaks(rec_hpf, peaks, method='center_of_mass', radius_um=50., **job_kwargs)
    print(peak_locations)

    # check for drifts
    # fs = rec_hpf.sampling_frequency
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.scatter(peaks['sample_ind'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)
    # @TODO: I think the above assumes peaks to be an Xarray?

    # we can also use the peak location estimates to have an insight of cluster separation before sorting
    fig, ax = plt.subplots(figsize=(15, 10))
    # si.plot_probe_map(raw_dat, ax=ax, with_channel_ids=True)
    plot_probe_group(raw_dat.get_probegroup(), same_axes=True, ax=ax)
    # ax.set_ylim(-100, 150)
    ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
    plt.savefig(base_folder / f'voltage_peaks_on_electrodes_of_probe{probe_num}.png', dpi=100)


if RUN_SORTING:
    sorting_folder.mkdir(exist_ok=True)
    sorter_algorithm = 'kilosort4'
    si.get_default_sorter_params(sorter_algorithm)

    if USE_REC:
        if AGGREGATE_SORTING:
            print('Using aggregate sorting!', flush=True)
            sorting = si.run_sorter_by_property(
                sorter_name=sorter_algorithm,
                recording=recording,
                grouping_property='group',
                working_folder=sorting_folder,
                verbose=True,
            )
        else:   
            split_preprocessed_recording = raw_dat.split_by("group")
            for group, sub_recording in split_preprocessed_recording.items():
                sorting = si.run_sorter(
                    sorter_name=sorter_algorithm,
                    recording=sub_recording,
                    output_folder=sorting_folder/f"{group}",
                    verbose=True,
                    remove_existing_folder=False,
                    )
                binary_path = sorting_folder/f"{group}"/"sorter_output"
                si.write_binary_recording(sub_recording, file_paths=binary_path / "recording.dat", dtype='int16', **job_kwargs)
                with open(binary_path / "params.py", "r") as params_file:
                    lines = params_file.readlines()
                lines[0] = "dat_path = 'recording.dat'\n"
                with open(binary_path / "params.py", "w") as params_file:
                    params_file.writelines(lines)
                    
    else:
        si.run_sorter(sorter_name=sorter_algorithm,
                                recording=recording,
                                output_folder=sorting_folder / f'{probe_num-1}',
                                docker_image=False,
                                verbose=True)


for probe_num in range(1, len(recording.get_probes())+1):
    print(f'Loading sorted data for probe {probe_num}...')
    sorting = si.read_sorter_folder(sorting_folder / f'{probe_num-1}')
    probe_folder = sorting_folder / f'{probe_num - 1}'

    if RUN_ANALYSIS:
        """
        POST SORTING
        https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html
        
        Contamination/'false positive'/'type I' metrics (amount of noise):
            * ISI violations (Allen default < 0.5): assuming <1.5ms refractory period interval
                The fraction that this gives means that contaminating spikes are occuring at
                a rate with that ratio relative to the "true" spikes.
            - sliding refractory period violations (sliding_rp_violation)
            - signal-to-noise ratio (SNR): ratio of the maximum amplitude of the mean spike waveform 
                to the standard deviation of the background noise on one channel.
                Allen recommend NOT to use it for NPix, because it only uses a single channel.
                Doesn't account for drift.
            - Nearest Neighbor hit rate (NN-hit rate; 0-1; e.g. >0.9): 
                looks at the PCs for one unit and calculates the fraction of their nearest 
                neighbors that fall within the same cluster. Negatively impacted by drift.
            
        Completeness/'false negative'/'type II' metrics (missing spikes):
            * presence ratio (0-0.99, it shouldn't reach 1, Allen default > 0.9), 
                  maybe reduce the bin size of presence ratio (default is 60s)
            * amplitude cutoff: percentage of distribution that is truncated (Allen default < 0.1)
            - NN-miss rate
            
        Drift metrics (changes in waveform due to failed tracking across electrode)
        
        PCA metrics:
            - isolation distance: the size of the PC sphere that includes as many 
                "other" spikes as are contained in the original unit's cluster.
                E.g., > 50 (larger means more isolated/less contaminated).
                Degrades with drift; value depends on #PCs.
            - L-ratio
            - D-prime: uses linear discriminant analysis to calculate the separability 
                of one unit's PC cluster and all of the others.
                Higher value means better isolation of cluster.
            - Silhouette score
            - NN-metrics (hit rate, miss rate, isolation, overlap)
        
        Some of these take a very long time, so we should try to only calculate those that we need.
        
        ALL measures require calculating the waveforms, templates, noise_levels
        
        Allen uses only isi_violations, amplitude_cutoff, and presence_ratio.
        https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
        
        The default bin size for presence_ratio is 60s, which I think is too large and reduced to 1s.
        """
        print('Creating analyzer...')
        analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True, format="memory")
        # analyzer_saved = analyzer.save_as(folder=probe_folder /  "analyzer", format="binary_folder")

        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500) # fast
        analyzer.compute("noise_levels")  # fast
        analyzer.compute("waveforms",  ms_before=1.5, ms_after=2., **job_kwargs) # slow, requires random_spikes
        
            
        analyzer.compute("templates", operators=["average", "median", "std"]) # fast, requires waveforms
        """"At this point, can calculate 
        metric_names=['firing_rate', 'presence_ratio', 'amplitude_cutoff', 
                      'snr', 'isi_violation', 'sliding_rp_violation']
        """
        analyzer.compute("unit_locations")  # requires templates, fast
        analyzer.compute("template_similarity")  # requires templates, fast
        
        metric_names = ['firing_rate', 'snr', 'isi_violation', 'amplitude_cutoff', 'sliding_rp_violation']
        metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)
        
        presence_ratio = si.compute_presence_ratios(analyzer, bin_duration_s=1)
        metrics['presence_ratio'] = pd.DataFrame(presence_ratio.values(), columns=['presence_ratio'], index=presence_ratio.keys())
        
        # --------------------------------------------------------------- #
        # shutil.rmtree(probe_folder/ "analyzer")
        # analyzer_saved = analyzer.save_as(folder=probe_folder /  "analyzer", format="binary_folder")
        # --------------------------------------------------------------- #
        
        """The following are slow. Only uncomment if necessary."""
        # analyzer.compute("spike_amplitudes", **job_kwargs)  # run in parallel using **job_kwargs
        # analyzer.compute("spike_locations")  # requires templates, very slow
        # analyzer.compute("correlograms")  # slow-ish

        # # --------------------------------------------------------------- #
        # shutil.rmtree(probe_folder/ "analyzer")
        # analyzer_saved = analyzer.save_as(folder=probe_folder /  "analyzer", format="binary_folder")
        # # --------------------------------------------------------------- #


        """
        Some metrics are based on PCA (like 'isolation_distance', 'l_ratio', 'd_prime') and 
        require to estimate PCA for their computation.
        At the moment, only nearest_neighbor seems attractive, but it is very costly to calculate.
        Have to first compute PCs, then project the waveforms, then calculate nearest neighbor metrics.
        """
        # analyzer.compute("principal_components") # medium speed
        # metrics = si.compute_quality_metrics(analyzer, metric_names=['nearest_neighbor'])
        
        
        # SortingAnalyzer can be saved to disk using save_as() which makes a copy of the analyzer and all computed extensions.
        # shutil.rmtree(probe_folder/ "analyzer")
        analyzer_saved = analyzer.save_as(folder=probe_folder /  "analyzer", format="binary_folder")
        print(analyzer_saved)
        
        # It is required to run sorting_analyzer.compute(input="spike_locations") first (if missing, values will be NaN)
        # drift_ptps, drift_stds, drift_mads = si.compute_drift_metrics(sorting_analyzer=analyzer)  # fast-ish
        # drift_ptps, drift_stds, and drift_mads are each a dict containing the unit IDs as keys,
        # and their metrics as values.

        # assert len(drift_ptps) == len(metrics)
        # metrics['drift_ptps'] = [drift_ptps[key] for key in np.arange(len(drift_ptps))]
        # # assert metrics['drift_ptps'][0] == drift_ptps[0]
        # metrics['drift_stds'] = [drift_stds[key] for key in np.arange(len(drift_stds))]
        # metrics['drift_mads'] = [drift_mads[key] for key in np.arange(len(drift_mads))]
        
        print(metrics)
    
        metrics.to_csv(probe_folder / "metrics")
    else:
        analyzer = si.load_sorting_analyzer(folder=probe_folder / "analyzer")
        metrics = pd.read_csv(probe_folder / "metrics", index_col=0)
    
    
    sorter_output_folder = probe_folder /  "sorter_output"
    
    for metric in ['firing_rate', 'presence_ratio', 'snr', 'isi_violations_count', 'isi_violations_ratio', 'amplitude_cutoff', 'sliding_rp_violation']:
        metric_df = pd.DataFrame()
        metric_df['cluster_id'] = metrics.index
        metric_df[metric] = metrics[metric]
        metric_df.to_csv(sorter_output_folder / ('cluster_' + metric + '.tsv'), sep='\t', index=False)
    
    
    if EXPORT_TO_PHY:
        # the export process is fast because everything is pre-computed
        si.export_to_phy(analyzer, output_folder=sorter_output_folder / 'phy', copy_binary=False, verbose=True)
        
    
    # Curation using metrics
    if FILTER_GOOD_UNITS:
        #A very common curation approach is to threshold these metrics to select good units:
    
        amplitude_cutoff_thresh = 0.1
        isi_violations_ratio_thresh = 1
        presence_ratio_thresh = 0.9
    
        our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
        print(our_query)
    
        # > (amplitude_cutoff < 0.1) & (isi_violations_ratio < 1) & (presence_ratio > 0.9)
    
        keep_units = metrics.query(our_query)
        keep_unit_ids = keep_units.index.values
        keep_unit_ids
        print(len(keep_unit_ids))
    
        # > array([ 7,  8,  9, 10, 12, 14])
    
    
        """Export final results to disk folder and visulize with sortingview"""
        # In order to export the final results we need to make a copy of the the waveforms, but only for the selected units (so we can avoid to compute them again).
        analyzer_clean = analyzer.select_units(keep_unit_ids, folder=probe_folder / 'analyzer_clean', format='binary_folder')
        analyzer_clean
    
        # > SortingAnalyzer: 383 channels - 6 units - 1 segments - binary_folder - sparse - has recording
        # > Loaded 9 extensions: random_spikes, waveforms, templates, noise_levels, correlograms, unit_locations, spike_amplitudes, template_similarity, quality_metrics
    
        #Then we export figures to a report folder
        # export spike sorting report to a folder
        si.export_report(analyzer_clean, probe_folder / 'report', format='png')
    
        # analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')
    # analyzer_clean


if ONLINE_CURATION:
    """
    Push the results to sortingview webased viewer
    1. At the conda prompt in the terminal: $ pip install kachery-cloud
    2. Then: $ kachery-cloud-init
    3. Link GitHub account to Kachery Cloud
    4. Run the line below, which will give a URL in the output
    """
    si.plot_sorting_summary(analyzer_clean, backend='sortingview')


print('Done.')
