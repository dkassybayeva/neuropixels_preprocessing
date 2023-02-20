%{
This file is the main preprocessing step to isolate
single unit recordings from Neuropixels using the output from Trodes

It runs the four main spike-sorting/clustering steps of Kilosort 2.5 or 3
1) Data preprocessing and channel selection
2) Drift calculation and batch reordering
3) The main optimization: clustering and template matching
4) Final merges and splits, then threshold detection

GK: Feb 20, 2023
%}


% ---------------------------------------------------------------------- %
%           Session paths: change prior to execution
% ---------------------------------------------------------------------- %
%WARNING deletes all files in data folder except for Trodes-specific files
delete_previous_KS_run = false;
remove_duplicates = false;

% the raw data binary file is in this folder CANT HAVE TRAILING SLASH
rootZ = 'X:\NeuroData\Nina2\20210623_121426.rec'; 
% path to temporary binary file (same size as data, should be on fast SSD)
rootH = 'X:\NeuroData\Nina2\20210623_121426.rec'; 
probenum = '1';
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%            Program paths: change upon new installation
% ---------------------------------------------------------------------- %
KS_version = '3.0';  % 2.5 or 3.0 (4 coming soon hopefully!)

docs_path = 'C:\Users\science person\Documents\';
preprocessing_path = strcat(docs_path,'MATLAB\neuropixels_preprocessing\Kilosort\');
sortingQuality_path = strcat(preprocessing_path, 'sorting_quality');

addpath(sortingQuality_path)
addpath(strcat(sortingQuality_path, '\core'))
addpath(strcat(sortingQuality_path, '\helpers'))

% path to kilosort folder
addpath(genpath(strcat(docs_path, 'MATLAB\Kilosort-', KS_version))) 

% for converting to Phy, https://github.com/kwikteam/npy-matlab
addpath(strcat(docs_path, 'npy-matlab-master')) 
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%               Setup and Load Kilosort Config File
% ---------------------------------------------------------------------- %
% take from Github folder and put it somewhere else (with master_file)
pathToYourConfigFile = strcat(preprocessing_path, 'configFiles'); 

% make chan map 
chanMapFile = 'channelMap.mat';
getChanMap(rootZ); %spikegadgets tool

% kilosort subfolder for KS output
[~,ss] = fileparts(rootZ);
kfolder = strcat(ss,'.kilosort', KS_version, '_probe', probenum);
if ~isfolder(fullfile(rootH,kfolder)),mkdir(fullfile(rootH,kfolder)), end
if ~isfolder(fullfile(rootZ,kfolder)),mkdir(fullfile(rootZ,kfolder)), end

% kilosort subfolder containing Trodes binary data file (Trodes will make a
% subfolder when exporting binary data file for kilosort)
ksdatafolder = strcat(ss,'.kilosort');

% binary dat file
kfile = strcat(ss,'.probe', probenum,'.dat');

% make config struct
if delete_previous_KS_run
   delete_KS_files(fullfile(rootZ,kfolder));
   delete_KS_files(fullfile(rootH,kfolder));
end

% make a copy of this script for reference
scriptpath = mfilename('fullpath'); scriptpath= [scriptpath,'.m'];
[~,scriptname]=fileparts(scriptpath); scriptname= [scriptname,'.m'];
ff = fullfile(rootZ,kfolder,scriptname);
ops.main_kilosort_script = ff;
copyfile(scriptpath,ff);
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                         Kilosort Parameters
% ---------------------------------------------------------------------- %
run(fullfile(pathToYourConfigFile, 'StandardConfig_384Kepecs.m'))

ops.trange    = [0 Inf]; % time range to sort
ops.NchanTOT  = 384; % total number of channels in your recording

% proc file on a fast SSD
ops.fproc   = fullfile(rootH, kfolder, 'temp_wh.dat'); 
ops.chanMap = fullfile(rootZ, chanMapFile);

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively

%   Blocks for registration (Replaces "datashift" option)
%   0 turns it off
%   1 does rigid registration
%   5 is default set by KS
ops.nblocks    = 5; 

% main parameter changes from Kilosort2.5 to v3.0 - []
ops.Th       = [9 9];

ops.fbinary = fullfile(rootZ, ksdatafolder, kfile);
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                      Run Kilosort Algorithm
% ---------------------------------------------------------------------- %
% 1) preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

% NEW STEP TO DO DATA REGISTRATION
% last input is for shifting data
rez = datashift2(rez, 1);

if strcmp(KS_version, '2.5')
    % ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
    iseed = 1;
                     
    % 2) and 3) main tracking and template matching algorithm
    rez = learnAndSolve8b(rez, iseed);
    % check_rez(rez);
    
    % OPTIONAL: remove double-counted spikes - solves issue in which 
    % individual spikes are assigned to multiple templates.
    % See issue 29: https://github.com/MouseLand/Kilosort/issues/29
    if remove_duplicates
        rez = remove_ks2_duplicate_spikes(rez);
    end
    
    % 4a) final merges
    rez = find_merges(rez, 1);
    % check_rez(rez);
    
    % 4b) final splits by SVD
    rez = splitAllClusters(rez, 1);
    % check_rez(rez);
    
    % 4c) decide on cutoff
    rez = set_cutoff(rez);
    % check_rez(rez);
    
    % 4d) eliminate widely spread waveforms (likely noise)
    rez.good = get_good_units(rez);
    
elseif strcmp(KS_version, '3.0')
    [rez, st3, tF]     = extract_spikes(rez);
    rez                = template_learning(rez, tF, st3);
    [rez, st3, tF]     = trackAndSort(rez);
    rez                = final_clustering(rez, tF, st3);
    rez                = find_merges(rez, 1);
end
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                       Report and Save Results
% ---------------------------------------------------------------------- %
fprintf('found %d good units \n', sum(rez.good>0))

fprintf('Saving results to Phy \n')
rezToPhy(rez, fullfile(rootH,kfolder));
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                       Compute Quality Metrics
% ---------------------------------------------------------------------- %
[cids, uQ, cR, isiV, histC] = sqKilosort.computeAllMeasures(fullfile(rootH, kfolder));

%save them for phy
sqKilosort.metricsToPhy(fullfile(rootH, kfolder), cids, uQ, isiV, cR, histC);
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%               Save the results to a Matlab file
% ---------------------------------------------------------------------- %

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% final time sorting of spikes, for apps that use st3 directly
[~, isort]   = sortrows(rez.st3);
rez.st3      = rez.st3(isort, :);

% Ensure all GPU arrays are transferred to CPU side before saving to .mat
rez_fields = fieldnames(rez);
for i = 1:numel(rez_fields)
    field_name = rez_fields{i};
    if(isa(rez.(field_name), 'gpuArray'))
        rez.(field_name) = gather(rez.(field_name));
    end
end

% save index times for spike number in extra mat file (since rez2.mat is
% superlarge & slow)
spikeTimes = uint64(rez.st3(:,1));
fname = fullfile(rootH,kfolder, 'spike_times.mat');
save(fname, 'spikeTimes', '-v7.3');

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(rootH,kfolder, 'rez2.mat');
save(fname, 'rez', '-v7.3');

%save KS figures
fname = fullfile(rootH,kfolder);
figHandles = get(0, 'Children');  
saveFigPNG(fname,figHandles(end-2:end));