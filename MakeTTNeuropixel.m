%{
Make spike time vectors for single unit recordings 
with Neuropixels & Trodes

convert KS2.5 clustering results, cured in Phy, to spike times 
using Trode's timestamps

author: TO Jan-Dec 2021
Updated by GK: Okt 6, 2022
%}

probenum = 2;
sf = 30000.0;  % sampling frequency
threshold = .001;  % flag any sampling gaps > 1ms

%----------------------------------------------------------------------%
%                           PATHS
%----------------------------------------------------------------------%
DATAPATH = 'X:\Neurodata';
rat = 'Nina2';
folder = '20210623_121426.rec';
session = '20210623_121426';
kfolder = [char(".kilosort_probe" + string(probenum))];

rec_file_path = fullfile(DATAPATH,rat,folder);
session_path = fullfile(rec_file_path, [session kfolder]);

% for using multiple probe shanks (NOT AUTOMATICALLY IMPLEMENTED!)
shank = str2double(kfolder(end)); 
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
% file pointer to raw/filtered data for waveform extraction
% this requires some parameters that might change!
% only works if file and KS channel map has 384 channels, 
% otherwise need to change a few things here
%----------------------------------------------------------------------%
nChInFile = 384;  
dataType = 'int16'; % data type of raw data file
BytesPerSample = 2;

rawfilename = fullfile(session_path, 'temp_wh.dat');
TotalBytes = get_file_size(rawfilename);
nSamp = TotalBytes/nChInFile/BytesPerSample;

mmf = memmapfile(rawfilename,'Format',{dataType, [nChInFile, nSamp], 'x'});
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                   Load Kilosort/Phy Cluster Data
%----------------------------------------------------------------------%
% Phy's clustering results have to be converted to mat file before
% (cf convert_spikes.py)
PhySpikes = load(fullfile(session_path,'spikes_per_cluster.mat'));

% KS cluster id per spike
SpikeCluster = readNPY(fullfile(session_path,'spike_clusters.npy'));

% Phy curing table
PhyLabels = tdfread(fullfile(session_path,'cluster_info.tsv'));

% load KS timestamps (these are indices in reality!) for each spike index
KSspiketimes = load(fullfile(session_path,'spike_times.mat')); 
KSspiketimes = KSspiketimes.spikeTimes;
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%              Load Trodes Times for Relative Timekeeping
%----------------------------------------------------------------------%
% load Trodes timestamps - in the general kilosort folder
kilosort_path = fullfile(rec_file_path,[session,'.kilosort']);
time_file = fullfile(kilosort_path, [session, '.timestamps.dat']);
Ttime = readTrodesExtractedDataFile(time_file); % >1GB var (3h recording)
Trodestimestamps = Ttime.fields.data;  % >1GB variable for a 3h recording
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%         get index of good units and save *time* of spies
%----------------------------------------------------------------------%
good_idx = find(all((PhyLabels.group(:,1:4)=='good'),2)); % row # of 'good'
good = PhyLabels.cluster_id( good_idx ); %Phy cluster_id labelled as 'good'

cellbase_dir = fullfile(session_path,'cellbase');
if ~isfolder(cellbase_dir)
    mkdir(cellbase_dir)
end

% create new, empty column in PhyLabels
PhyLabels.cellbase_name = cell(length(PhyLabels.cluster_id), 1);
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                       Create Spike Time Vectors
%----------------------------------------------------------------------%
for k =1:length(good)
    %-----------------------Spikes-------------------------------%
    clu = good(k);
    Sind = PhySpikes.(['f',num2str(clu)]);  % spike index per cluster
    KStime = KSspiketimes(Sind+1);  % spike index to time index

    % time index to time in Trodes format (this accounts for potential 
    % lost packages/data in Trodes)
    SpikeTimes = Trodestimestamps(KStime+1); 
    
    % actual spike times in seconds
    % Trodes saves timestamp as index in sampling frequency
    TS = double(SpikeTimes)/sf; 
    
    % save in cellbase format - one mat file per unit (cellbase convention)
    % cellbase convention: count ntrode/probe and unit within ntrode/probe
    unitname = strcat(num2str(shank),'_',num2str(k)); 
    fname = fullfile(cellbase_dir, ['TT',unitname,'.mat']);
    save(fname,'TS');
    

    %------------------------------------------------------------%
    %save cellbase name to Phy labels for future provenance
    PhyLabels.cellbase_name{good_idx(k)} = unitname; 
    

    %-------------------extract spike waveforms------------------%
    % spike times for specific cluster
    theseST = KSspiketimes(SpikeCluster==clu)+1; 
    
    % extract at most the first 100 spikes
    extractST = theseST(10:min(110,length(theseST)));

    nWFsToLoad = length(extractST);

    % samples around the spike times to load
    wfWin = [-20:40];  
    nWFsamps = length(wfWin);
    theseWF = zeros(nWFsToLoad, nChInFile, nWFsamps);

    for i=1:nWFsToLoad
        win_begin = extractST(i)+wfWin(1);
        win_end = extractST(i)+wfWin(end);
        tempWF = mmf.Data.x(1:nChInFile, win_begin:win_end);
        theseWF(i,:,:) = tempWF;
    end

    % average spikes
    WFm=squeeze(mean(theseWF,1));

    % find maximum channel
    [~,midx]=max(max(WFm,[],2));
    WF=WFm(midx,:);
    fname = fullfile(cellbase_dir, ['WF', unitname, '.mat']);
    save(fname, 'WF');    
end
%----------------------------------------------------------------------%

%----------------------------------------------------------------------%
%                               Gaps
%----------------------------------------------------------------------%
gaps = double(diff(Trodestimestamps))/double(sf); %> threshold;

% gaps_ts should be the difference the timestamp where gaps *starts*
gaps_ts = double(Trodestimestamps(gaps > threshold)) / double(sf);
gaps = gaps(gaps > threshold);
gaps_ts = gaps_ts(gaps >threshold);

% also save some info for later in cellbase folder
fname_gaps = fullfile(cellbase_dir, 'GAPS.mat');
save(fname_gaps, 'gaps', 'gaps_ts')
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                         Save Related Info
%----------------------------------------------------------------------%
% save cluster quality metrics
fname = fullfile(cellbase_dir, ['PhyLabels_',num2str(shank),'.mat']);
save(fname,'PhyLabels');

% save TRODES analog input converted to TTL event
fname = fullfile(DATAPATH,rat,folder,[session,kfolder]);
[Events_TTL, Events_TS] = extractTTLs(fname,fname_gaps);
save(fullfile(cellbase_dir,'EVENTS.mat'),'Events_TS', 'Events_TTL');

