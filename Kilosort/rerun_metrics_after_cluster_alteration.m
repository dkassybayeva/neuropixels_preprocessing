%{ 
Script to rerun quality metrics in SortingQuality/sqKilosort after
manual validation in Phy, where merges and splits may have occurred, 
resulting in new clusters needing quality metrics.

author: Greg Knoll
email: gregory@bccn-berlin.de
date: Okt 6, 2022
%}   

KS_version = '2.5';  % 2.5 or 3.0 (4 coming soon hopefully!)

% the paths should be the same as in the main_kilosort*.m file run before using Phy
probenum = '1';

% directory of binary file
rootH = 'Y:\NeuroData\Nina2\20210625_114657.rec';

%kilosort subfolder for KS output
[~,ss] = fileparts(rootH);
ks_output_dir = strcat(ss,'.kilosort', KS_version, '_probe', probenum);

if isfolder(fullfile(rootH, ks_output_dir)) % only run for existing data
    
% compute quality metrics
[cids, uQ, cR, isiV, histC] = sqKilosort.computeAllMeasures(fullfile(rootH, ks_output_dir));

%save them for phy
sqKilosort.metricsToPhy(fullfile(rootH, ks_output_dir), cids, uQ, isiV, cR, histC);

disp('Done.')

end
