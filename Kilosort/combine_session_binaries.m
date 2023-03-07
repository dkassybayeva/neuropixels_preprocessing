%{
Combine the .rec binary outputs from multiple Trodes sessions.

Author: GK (extracted from previous "batch" script)
Date: March 2023
%}

clear all
close all
clc

% path to kilosort folder
% KS_version = '2.5';  % 2.5 or 3.0 (4 coming soon hopefully!)
% addpath(genpath(strcat(docs_path, 'MATLAB\Kilosort-', KS_version))) 

% addpath('C:\Users\Adam\Documents\npy-matlab') % for converting to Phy
% addpath('C:\Users\Adam\Documents\MATLAB\neuropixels_preprocessing\sortingQuality')

% raw binary dat files to combine (SSD or HHD)
base_dir = {'D:\NeuroData\Nina2\', 'Y:\NeuroData\Nina2\'};
sessions = {'20210623_121426', '20210625_114657'};
probe_num = '1';

% Create a list of paths using the components above
session_dat_folders = cell(numel(sessions), 1);
session_dates = cell(numel(sessions), 1);
for sesh = 1:numel(sessions)
    rec_dir = strcat(base_dir(sesh), sessions(sesh), '.rec');
    ks_dir = strcat(sessions(sesh), '.kilosort');
    probe_dat = strcat(sessions(sesh), '.probe', probe_num, '.dat');
    session_dat_folders(sesh) = fullfile(rec_dir, ks_dir, probe_dat);
    session_date = strsplit(char(sessions(sesh)), '_');
    session_dates(sesh) = session_date(1);
end


% Output folder for merged binary (SSD or HDD)
multi_session_folder = fullfile('X:\Neurodata\Nina2', strjoin(session_dates, '_'));
combined_dat_folder = fullfile(multi_session_folder, strcat('probe', probe_num));
if ~isfolder(combined_dat_folder), mkdir(combined_dat_folder), end

% copy one chan map to output folder (assumes all session chan maps are the same)
first_session_dir = strcat(base_dir{1}, sessions{1}, '.rec');
copyfile(fullfile(first_session_dir, 'channelMap.mat'), ...
         fullfile(multi_session_folder,'channelMap.mat'));
% ks_output_folder = 'D:\Neurodata\TQ03\20210616_20210618_2'; % saves large temp file and small(er) output files (SSD)



%make config file
docs_path = 'C:\Users\science person\Documents\';
preprocessing_path = strcat(docs_path,'MATLAB\neuropixels_preprocessing\Kilosort\');
config_file = fullfile([preprocessing_path, 'configFiles'], 'StandardConfig_384Kepecs.m');
run(config_file)
% chanMapFile = 'neuropixPhase3B1_kilosortChanMapTORBEN.mat';
% chanMapFile = 'channelMap.mat';

%make output folders
% if ~isfolder(ks_output_folder),mkdir(ks_output_folder), end
   
%combined binary data file
ops.session_dat_folders = session_dat_folders;
ops.combined_dat_folder = combined_dat_folder;
% ops.ks_output_folder = ks_output_folder;
ops.trange    = [0 Inf];  % time range to sort
ops.NchanTOT  = 384;  % total number of channels in your recording
ops.fbinary = fullfile(combined_dat_folder, 'combined.mat');



if exist(ops.fbinary,'file') ~= 2
    ops = concat_dat_files(ops, session_dat_folders, ops.fbinary, false);
    %make combined data file
%else
   % [~,ss] = fileparts(session_dat_folders{1});
  %  ksdatafolder = strcat(ss,'.kilosort');
  %  kfile = strcat(ss,'.probe1.dat');
  %  datafile = fullfile(session_dat_folders{1}, ksdatafolder, kfile);
    
  %  bytes       = get_file_size(datafile); % size in bytes of raw binary
  %  NT       = ops.NT ; % number of timepoints per batch
  %  NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc
 
  %  nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
   % ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
  %  ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
  %  ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
  %  ops.midpoint      = ceil(ops.sampsToRead /NT); %number of bacthes in first file
end

%save ops
save(fullfile(combined_dat_folder, 'combined_ops.mat'), 'ops', '-v7.3');
