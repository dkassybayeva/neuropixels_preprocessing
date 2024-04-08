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
from xml.etree import ElementTree


base_folder = Path('Y:/NeuroData/TQ03/TQ03_20210617_combined.kilosort/')
# base_folder = Path('/home/gerg/Workspace/ott_neuropix_data/TQ03/ephys/TQ03_20210617_combined.kilosort')

USE_REC = False
FILTER_RAW_BEFORE_SORTING = True  # applies HPF and CMR
SAVE_PREPROCESSING = False

PLOT_BIG_HEATMAPS = False
PLOT_SOME_CHANNELS = False
PLOT_NOISE = False
PLOT_PEAKS_ON_ELECTRODES = False

if USE_REC:
    rec_file = base_folder / '20210617_114801.rec'
    
    
    # stream_names, stream_ids = si.get_neo_streams('spikegadgets', rec_file)
    # print('Available streams', stream_ids)
    
    # sg_rec = SpikeGadgetsRecordingExtractor(file_path=rec_file, stream_id='trodes')
    # tvec = sg_rec.get_times()
    # channels = sg_rec.get_channel_ids()
    # duration_s = sg_rec.get_total_duration()
    # sg_rec.get_probe().to_dataframe()
    raw_dat = read_spikegadgets(rec_file)
    
    """
    Check that imported data from SpikeInterface for a single channel is the same as that written by Trodes exporter
    """
    fs = raw_dat.get_sampling_frequency()
    n_samples = raw_dat.neo_reader._raw_memmap.shape[0]
    first_trace = raw_dat.get_traces(segment_index=0, channel_ids=['735'], start_frame=0, end_frame=1000).flatten()
    first_trace.shape
    
    """
    GK note: each channel in the .dat has a corresponding channel in the raw_dat import.
    However, there are two issues:
    - the channels in the .dat don't seem to have a logical correspondence with those in the header
    - the .dat is -1 * the values in the .rec
    
    'trode1384chan735', 'trode1383chan734', 'trode1382chan671',
    'trode2384chan767', 'trode2383chan766', 'trode2382chan703',
    """
    
    temp = np.fromfile(base_folder / '20210617_114801.kilosort' / '20210617_114801.probe1.dat', dtype='int16').reshape(384, -1, order='F')
    assert temp.shape[1] == n_samples
    first_trace_from_dat = temp[383][:1000]  # use Fortran / column reordering (default in numpy is C / row ordering)
    scaling = 0.018311105685598315
    # assert np.all(first_trace == -1 * first_trace_from_dat)
    # int(temp[0])
    
    # fig, ax = plt.subplots(figsize=(20, 10))
    # si.plot_traces({'pr1-ch1': raw_dat}, backend='matplotlib', mode='line', ax=ax, show_channel_ids=True, channel_ids=raw_dat.channel_ids[[0]], color='k', time_range=[0, 0.33], return_scaled=True)
    # plt.plot(np.linspace(0, 1000/fs, 1000), first_trace * scaling)
    plt.plot(first_trace * scaling)
    plt.plot(first_trace_from_dat * -1 * scaling)
    plt.ylabel('uV')
    plt.show()
    
    
    # probe1_rec = raw_dat.split_by('group')[0]
    # probe1 = raw_dat.get_probes()[0]
    # assert np.all(probe1_rec.channel_ids == probe1.channel_ids)
    
    # probe1.to_dataframe()
    
    # plot_probe(probe)
    plot_probe_group(raw_dat.get_probegroup(), same_axes=True)
    plt.show()
    # plot_probe_group(raw_dat.get_probegroup(), same_axes=False, with_contact_id=True)
    # plt.show()
else:
    from neuropixels_preprocessing.misc_utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
    from probeinterface import Probe
    CONTACT_WIDTH = 16  # um
    CONTACT_HEIGHT = 20  # um
    
    probe_num = 1
    
    # ------------------------------------ #
    #           Read in data
    # ------------------------------------ #
    binary_file = base_folder / f'TQ03_20210617_combined.probe{probe_num}.dat'
    raw_dat = si.read_binary(file_paths=binary_file, 
                             sampling_frequency=30_000., 
                             num_channels=384, 
                             dtype='int16')
    
    # ------------------------------------ #
    #           Construct Probe
    # ------------------------------------ #
    pad_coords_in_um = readTrodesExtractedDataFile(base_folder / f'TQ03_20210617_combined.channelmap_probe{probe_num}.dat')['data']
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

    plot_probe(probe)
    plt.show()



    


if FILTER_RAW_BEFORE_SORTING:
    rec_hpf = si.highpass_filter(raw_dat, freq_min=400.)
    
    
    # bad_channel_ids, channel_labels = si.detect_bad_channels(rec_hpf)
    # rec_hpf_good = rec_hpf.remove_channels(bad_channel_ids)
    # print('bad_channel_ids', bad_channel_ids)
    
    # rec_hpf_shift = si.phase_shift(rec_hpf)
    recording = si.common_reference(rec_hpf, operator="median", reference="global")
else:
    recording = raw_dat

# here we use static plot using matplotlib backend
if PLOT_BIG_HEATMAPS:
    if FILTER_RAW_BEFORE_SORTING:
        fig, axs = plt.subplots(ncols=3, figsize=(20, 10))
        
        si.plot_traces(raw_dat, backend='matplotlib', time_range=(4,6),  clim=(-4000, 4000), ax=axs[0])
        si.plot_traces(rec_hpf, backend='matplotlib', time_range=(4,6),  clim=(-4000, 4000), ax=axs[1])
        si.plot_traces(recording, backend='matplotlib',  clim=(-4000, 4000), ax=axs[2])
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
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
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



sorting_folder = base_folder / 'spike_interface_kilosort4_output'
sorting_folder.mkdir(exist_ok=True)
# check default params for kilosort4
si.get_default_sorter_params('kilosort4')


if USE_REC:
    """For single probes"""
    # sorting = si.run_sorter('kilosort4', raw_dat, grouping_property='group', output_folder=base_folder / 'kilosort4_output', docker_image=False, verbose=True)
    # alternative:
    # sorting = si.run_sorter('kilosort4', raw_dat.split_by('group')[0], output_folder=base_folder / 'kilosort4_output', docker_image=False, verbose=True)
    # print(sorting)
    
    """For multiple probes"""
    # -----Either split ahead of time
    
    # split_preprocessed_recording = raw_dat.split_by("group")
    # sortings = {}
    # for group, sub_recording in split_preprocessed_recording.items():
    #     sorting = si.run_sorter(
    #         sorter_name='kilosort4',
    #         recording=split_preprocessed_recording,
    #         output_folder=f"folder_KS2_group{group}"
    #         )
    #     sortings[group] = sorting
    
    # -----Or use aggregate sorting
    aggregate_sorting = si.run_sorter_by_property(
        sorter_name='kilosort4',
        recording=recording,
        grouping_property='group',
        working_folder=sorting_folder
    )
    print(aggregate_sorting)
    
    # the results can be read back for futur session
    sorting = si.read_sorter_folder(sorting_folder)
else:
    # from kilosort.utils import download_probes
    # download_probes()
    # from kilosort import run_kilosort
    # # NOTE: 'n_chan_bin' is a required setting, and should reflect the total number
    # #       of channels in the binary file. For information on other available
    # #       settings, see `kilosort.run_kilosort.default_settings`.
    # settings = {'filename':binary_file,'n_chan_bin': 384, 'results_dir': sorting_folder / f'probe{probe_num}'}

    # ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = \
    # run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat')
    
    
    sorting = si.run_sorter(sorter_name='kilosort4',
                            recording=recording,
                            output_folder=sorting_folder / f'probe{probe_num}', 
                            docker_image=False, 
                            verbose=True)
    sorting = si.read_sorter_folder(sorting_folder / f'probe{probe_num}')

"""
POST SORTING
"""

analyzer = si.create_sorting_analyzer(folder=sorting_folder / f'probe{probe_num}', sparse=True, format="memory")
# analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True, format="memory")
print(analyzer)

analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms",  ms_before=1.5,ms_after=2., **job_kwargs)
analyzer.compute("templates", operators=["average", "median", "std"])
analyzer.compute("noise_levels")
print(analyzer)

analyzer.compute("correlograms")
analyzer.compute("unit_locations")
analyzer.compute("spike_amplitudes", **job_kwargs)  # run in parallel using **job_kwargs
analyzer.compute("template_similarity")
print(analyzer)

# SortingAnalyzer can be saved to disk using save_as() which make a copy of the analyzer and all computed extensions.
analyzer_saved = analyzer.save_as(folder=sorting_folder / "analyzer", format="binary_folder")
print(analyzer_saved)


"""
Quality Metrics
"""
# Some metrics are based on PCA (like 'isolation_distance', 'l_ratio', 'd_prime') and require to estimate PCA for their computation. This can be achieved with:
analyzer.compute("principal_components")

metrics = analyzer.compute("quality_metrics").get_data()
# equivalent to
# metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']
# metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)
print(metrics)

metrics.to_csv(sorting_folder / "metrics")

print('Done.')