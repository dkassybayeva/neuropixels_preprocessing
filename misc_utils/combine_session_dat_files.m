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
session_dates = {'20210616', '20210617'};
probe_num = '1';
dat_files = {'X:\Neurodata\Nina2\ephys\20210625_114657.rec\20210625_114657.kilosort\20210625_114657.probe2.dat', 
             'Y:\Neurodata\Nina2\ephys\20210626_153916.rec\20210626_153916.kilosort\20210626_153916.probe2.dat'};
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%                           Output Paths
% ------------------------------------------------------------------- %
% Output folder for merged binary (SSD or HDD)
multi_session_folder = fullfile('X:\Neurodata\Nina2\ephys\', strjoin(session_dates, '_'));
combined_dat_folder = fullfile(multi_session_folder, strcat('probe', probe_num));
if ~isfolder(combined_dat_folder), mkdir(combined_dat_folder), end
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%                           Parameters
% ------------------------------------------------------------------- %
ops.session_dat_folders = dat_files;
ops.combined_dat_folder = combined_dat_folder;
ops.NT        = 65600;
ops.fs        = 30000;
ops.trange    = [0 Inf];  % time range to sort
ops.NchanTOT  = 384;      % total number of channels in your recording
ops.fbinary = fullfile(combined_dat_folder, 'combined.dat');
% ------------------------------------------------------------------- %


% ------------------------------------------------------------------- %
%         Combine the data files and save combined parameters
% ------------------------------------------------------------------- %
if exist(ops.fbinary,'file') ~= 2  % don't overwrite previous attempts
    ops = concat_dat_files(ops, dat_files, ops.fbinary, false);
end

save(fullfile(combined_dat_folder, 'combined_ops.mat'), 'ops', '-v7.3');

fprintf('\n');
disp('-----> End program.');