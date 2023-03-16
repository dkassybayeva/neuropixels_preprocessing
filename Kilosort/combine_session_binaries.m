%{
Combine the .rec binary outputs from multiple Trodes sessions.

Author: GK (extracted from previous "batch" script)
Date: March 2023
%}

clear all
close all
clc

% ------------------------------------------------------------------- %
%                           Input Paths
% ------------------------------------------------------------------- %
% raw binary dat files to combine (SSD or HHD)
base_dir = {'D:\NeuroData\Nina2\', 'D:\NeuroData\Nina2\'};
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
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%                           Output Paths
% ------------------------------------------------------------------- %
% Output folder for merged binary (SSD or HDD)
multi_session_folder = fullfile('X:\Neurodata\Nina2', strjoin(session_dates, '_'));
combined_dat_folder = fullfile(multi_session_folder, strcat('probe', probe_num));
if ~isfolder(combined_dat_folder), mkdir(combined_dat_folder), end
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%                           Parameters
% ------------------------------------------------------------------- %
% copy one chan map to output folder 
% (assumes all session chan maps are the same)
first_session_dir = strcat(base_dir{1}, sessions{1}, '.rec');
copyfile(fullfile(first_session_dir, 'channelMap.mat'), ...
         fullfile(multi_session_folder,'channelMap.mat'));

% Run config file to pull ops variable into Workspace
docs_path = 'C:\Users\science person\Documents\';
preprocessing_path = strcat(docs_path,'MATLAB\neuropixels_preprocessing\Kilosort\');
config_file = fullfile([preprocessing_path, 'configFiles'], 'StandardConfig_384Kepecs.m');
run(config_file)

%combined binary data file
ops.session_dat_folders = session_dat_folders;
ops.combined_dat_folder = combined_dat_folder;
% ops.ks_output_folder = ks_output_folder;
ops.trange    = [0 Inf];  % time range to sort
ops.NchanTOT  = 384;  % total number of channels in your recording
ops.fbinary = fullfile(combined_dat_folder, 'combined.mat');
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%         Combine the data files and save combined parameters
% ------------------------------------------------------------------- %
if exist(ops.fbinary,'file') ~= 2  % don't overwrite previous attempts
    ops = concat_dat_files(ops, session_dat_folders, ops.fbinary, false);
end

save(fullfile(combined_dat_folder, 'combined_ops.mat'), 'ops', '-v7.3');
