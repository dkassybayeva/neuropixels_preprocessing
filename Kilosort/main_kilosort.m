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

clear
close all
clc

% ---------------------------------------------------------------------- %
%           Session paths: change prior to execution
% ---------------------------------------------------------------------- %
%WARNING deletes all files in data folder except for Trodes-specific files
delete_previous_KS_run = false;
remove_duplicates = true;

% the raw data binary file is in this folder CANT HAVE TRAILING SLASH
raw_data_dir = 'Y:\NeuroData\Nina2\20210625_114657.rec';

% path to temporary binary file (same size as data, should be on fast SSD)
output_base = 'Y:\NeuroData\Nina2\20210625_114657.rec';
probenum = '1';
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%            Program paths: change upon new installation
% ---------------------------------------------------------------------- %
KS_version = '2.5';  % 2.5 or 3.0 (4 coming soon hopefully!)

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
config_file = fullfile([preprocessing_path, 'configFiles'], 'StandardConfig_384Kepecs.m');

% make chan map 
getChanMap(raw_data_dir);  % spikegadgets tool that creates channelMap.mat

% kilosort subfolder containing Trodes binary data file (Trodes will make a
% subfolder when exporting binary data file for kilosort)
[~,ss] = fileparts(raw_data_dir);
ks_data_folder = strcat(ss, '.kilosort');
input_dir = fullfile(raw_data_dir, ks_data_folder);
if ~isfolder(input_dir)
    error('.kilosort directory not found. It should be exported from Trodes.')
end

% binary dat file
probe_binary_dat = strcat(ss, '.probe', probenum, '.dat');

% subfolder for Kilosort output
ks_output_folder = strcat(ss, '.kilosort', KS_version, '_probe', probenum);
output_dir = fullfile(output_base, ks_output_folder);
if ~isfolder(output_dir), mkdir(output_dir), end

% make config struct
if delete_previous_KS_run
   delete_KS_files(input_dir);
   delete_KS_files(output_dir);
end

% make a copy of this script for reference
curr_script_path = [mfilename('fullpath'),'.m'];
[~, scriptname] = fileparts(curr_script_path);

% point Kilosort to copied script
ops.main_kilosort_script = fullfile(output_dir, [scriptname, '.m']);
copyfile(curr_script_path, ops.main_kilosort_script);
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                         Kilosort Parameters
% ---------------------------------------------------------------------- %
run(config_file)

ops.trange    = [0 Inf]; % time range to sort
ops.NchanTOT  = 384; % total number of channels in your recording

% proc file on a fast SSD
ops.fproc   = fullfile(output_dir, 'temp_wh.dat'); 
ops.chanMap = fullfile(raw_data_dir, 'channelMap.mat');

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively

%   Blocks for registration (Replaces "datashift" option)
%   0 turns it off
%   1 does rigid registration
%   5 is default set by KS
ops.nblocks    = 5; 

% main parameter changes from Kilosort2.5 to v3.0 - default is [10 4]
ops.Th       = [10 4];

ops.fbinary = fullfile(input_dir, probe_binary_dat);
% ---------------------------------------------------------------------- %
ops  % prints parameters


% ---------------------------------------------------------------------- %
%                      Run Kilosort Algorithm
% ---------------------------------------------------------------------- %
% 1) preprocess data to create temp_wh.dat
disp('------------------------------------------------------');
disp('preprocessDataSub');
disp('------------------------------------------------------');
rez = preprocessDataSub(ops);

% NEW STEP TO DO DATA REGISTRATION
% last input is for shifting data
disp('------------------------------------------------------');
disp('datashift2');
disp('------------------------------------------------------');
rez = datashift2(rez, 1);

if strcmp(KS_version, '2.5')
    % ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
    iseed = 1;
                     
    disp('------------------------------------------------------');
    disp('learnAndSolve8b');
    disp('------------------------------------------------------');
    % 2) and 3) main tracking and template matching algorithm
    rez = learnAndSolve8b(rez, iseed);
    % check_rez(rez);
    
    % OPTIONAL: remove double-counted spikes - solves issue in which 
    % individual spikes are assigned to multiple templates.
    % See issue 29: https://github.com/MouseLand/Kilosort/issues/29
    disp('------------------------------------------------------');
    disp('remove_ks2_duplicate_spikes');
    disp('------------------------------------------------------');
    if remove_duplicates
        rez = remove_ks2_duplicate_spikes(rez);
    end
    
    % 4a) final merges
    disp('------------------------------------------------------');
    disp('find_merges');
    disp('------------------------------------------------------');
    rez = find_merges(rez, 1);
    % check_rez(rez);
    
    % 4b) final splits by SVD
    disp('------------------------------------------------------');
    disp('splitAllClusters');
    disp('------------------------------------------------------');
    rez = splitAllClusters(rez, 1);
    % check_rez(rez);
    
    % 4c) decide on cutoff
    disp('------------------------------------------------------');
    disp('set_cutoff');
    disp('------------------------------------------------------');
    rez = set_cutoff(rez);
    % check_rez(rez);
    
    % 4d) eliminate widely spread waveforms (likely noise)
    disp('------------------------------------------------------');
    disp('get_good_units');
    disp('------------------------------------------------------');
    rez.good = get_good_units(rez);

elseif strcmp(KS_version, '3.0')
    disp('------------------------------------------------------');
    disp('extract_spikes');
    disp('------------------------------------------------------');
    [rez, st3, tF]     = extract_spikes(rez);
    disp('------------------------------------------------------');
    disp('template_learning');
    disp('------------------------------------------------------');
    rez                = template_learning(rez, tF, st3);
    disp('------------------------------------------------------');
    disp('trackAndSort');
    disp('------------------------------------------------------');
    [rez, st3, tF]     = trackAndSort(rez);
    disp('------------------------------------------------------');
    disp('final_clustering');
    disp('------------------------------------------------------');
    rez                = final_clustering(rez, tF, st3);
    disp('------------------------------------------------------');
    disp('final_merges');
    disp('------------------------------------------------------');
    rez                = find_merges(rez, 1);
end
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                       Report and Save Results
% ---------------------------------------------------------------------- %
fprintf('found %d good units \n', sum(rez.good>0))

fprintf('Saving results to Phy \n')
rezToPhy(rez, output_dir);
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                       Compute Quality Metrics
% ---------------------------------------------------------------------- %
[cids, uQ, cR, isiV, histC] = sqKilosort.computeAllMeasures(output_dir);

%save them for phy
sqKilosort.metricsToPhy(output_dir, cids, uQ, isiV, cR, histC);
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
save(fullfile(output_dir, 'spike_times.mat'), 'spikeTimes', '-v7.3');

% save final results as rez2
fprintf('Saving final results in rez2  \n')
save(fullfile(output_dir, 'rez2.mat'), 'rez', '-v7.3');

%save KS figures
figHandles = get(0, 'Children');  
saveFigPNG(output_dir, figHandles(end-2:end));