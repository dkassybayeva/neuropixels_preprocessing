%{
Make spike time vectors for single unit recordings 
with Neuropixels & Trodes

convert KS2.5 clustering results, cured in Phy, to spike times 
using Trode's timestamps

author: GK (from Matlab_pipeline/extract_spiketimes_and_gaps_and_waveforms.m)
date: March 2023
updated: May 2024
%}

probenum = 1;
sf = 30000.0;  % sampling frequency

n_waveforms = 100;
random_waveforms = true;

% samples around the spike times to load
waveform_win = [-20:40];  
waveform_len = length(waveform_win);

%----------------------------------------------------------------------%
%                           PATHS
%----------------------------------------------------------------------%
DATAPATH = 'Y:\Neurodata';
rat = 'TQ03';
session = '20210617_115450';
folder = strcat(session, '.rec');

rec_file_path = fullfile(DATAPATH, rat, 'ephys', folder);
sorting_path = fullfile(rec_file_path, [char("sorting_output\probe" + string(probenum) + "\sorter_output")]);

rawfilename = fullfile(rec_file_path, char(session + ".kilosort\" + session + ".probe" + string(probenum) + ".dat"));
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
% file pointer to raw/filtered data for waveform extraction
% this requires some parameters that might change!
% only works if file and KS channel map has 384 channels, 
% otherwise need to change a few things here
%----------------------------------------------------------------------%
n_channels = 384;  
dataType = 'int16'; % data type of raw data file
bytes_per_sample = 2;

total_bytes_in_file = get_file_size(rawfilename);
nSamp = total_bytes_in_file / n_channels / bytes_per_sample;

mmf = memmapfile(rawfilename, 'Format', {dataType, [n_channels, nSamp], 'x'});
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                   Load Kilosort/Phy Cluster Data
%----------------------------------------------------------------------%
% KS cluster id per spike
SpikeCluster = readNPY(fullfile(sorting_path,'spike_clusters.npy'));

% Phy curing table
PhyLabels = tdfread(fullfile(sorting_path,'cluster_info.tsv'));

% load KS timestamps (these are indices in reality!) for each spike index
KSspiketimes = readNPY(fullfile(sorting_path,'spike_times.npy'));
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%         get index of good units and save *time* of spikes
%----------------------------------------------------------------------%
good_idx = find(all((PhyLabels.group(:,1:4)=='good'),2)); % row # of 'good'
good = PhyLabels.cluster_id( good_idx ); %Phy cluster_id labelled as 'good'

waveform_dir = fullfile(rec_file_path, 'preprocessing_output', char("probe" + string(probenum)), 'waveforms');
if ~isfolder(waveform_dir)
    mkdir(waveform_dir)
end
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                       Create Spike Time Vectors
%----------------------------------------------------------------------%
for k =1:length(good)
    %-----------------------Spikes-------------------------------%
    clu = good(k);
    fprintf([num2str(k), ': cluster=', num2str(clu), '\n']);

    % save in cellbase format - one mat file per unit (cellbase convention)
    % cellbase convention: count ntrode/probe and unit within ntrode/probe
    unitname = strcat('unit_', num2str(clu));
    %------------------------------------------------------------%

    %------------------------------------------------------------%
    %       select spikes whose waveforms will be extracted      %
    %------------------------------------------------------------%
    % spike times for specific cluster
    cluster_spiketimes = KSspiketimes(SpikeCluster==clu)+1; 
    n_waveforms = min(n_waveforms, length(cluster_spiketimes));

    if random_waveforms
        waveform_idx = randi([1, length(cluster_spiketimes)], 1, n_waveforms);  
    else % extract the first spikes after the 10th
        waveform_idx = 10: 10 + n_waveforms; 
    end
    selected_spiketimes = cluster_spiketimes(waveform_idx);
    %------------------------------------------------------------%

    %------------------------------------------------------------%
    %               Put waveforms into a matrix
    %------------------------------------------------------------%
    wave_mat = zeros(n_waveforms, n_channels, waveform_len);

    for i=1:n_waveforms
        win_begin = selected_spiketimes(i) + waveform_win(1);
        win_end = selected_spiketimes(i) + waveform_win(end);

        wave_mat(i,:,:) = mmf.Data.x(1:n_channels, win_begin:win_end);
    end
    %------------------------------------------------------------%

    %------------------------------------------------------------%
    %       Only keep the waveform on the clearest channel
    %------------------------------------------------------------%
    % average spikes
    mean_waveform=squeeze(mean(wave_mat,1));

    % find maximum channel
    [~, loudest_ch] = max(max(mean_waveform,[],2));
    mean_waveform_max_ch = mean_waveform(loudest_ch,:);

    wave_mat = squeeze(wave_mat(:, loudest_ch, :));
    %------------------------------------------------------------%

    fname = fullfile(waveform_dir, [unitname, '.mat']);
    save(fname, 'wave_mat');    
end
%----------------------------------------------------------------------%
