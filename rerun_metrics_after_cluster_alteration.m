%{ 
Script to rerun quality metrics in SortingQuality/sqKilosort after
manual validation in Phy, where merges and splits may have occurred, 
resulting in new clusters needing quality metrics.

author: Greg Knoll
email: gregory@bccn-berlin.de
date: Okt 6, 2022
%}   

% the paths should be the same as in the main_kilosort*.m file run before using Phy
probenum = 2;

% directory of binary file
rootH = 'X:\NeuroData\Nina2\20210623_121426.rec';

%kilosort subfolder for KS output
[~,ss] = fileparts(rootH);
kfolder = strcat(ss,'.kilosort_probe',num2str(probenum));

if isfolder(fullfile(rootH,kfolder)) % only run for existing data
    
% compute quality metrics
[cids, uQ, cR, isiV, histC] = sqKilosort.computeAllMeasures(fullfile(rootH, kfolder));

%save them for phy
sqKilosort.metricsToPhy(fullfile(rootH, kfolder), cids, uQ, isiV, cR, histC);

end
