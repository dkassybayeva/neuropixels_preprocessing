function [cids, uQ, cR, isiV, histC] = computeAllMeasures(resultsDirectory)

    clusterPath = fullfile(resultsDirectory, 'cluster_groups.csv');
    spikeClustersPath = fullfile(resultsDirectory,'spike_clusters.npy');
    spikeTemplatesPath = fullfile(resultsDirectory,'spike_templates.npy');

    if exist(clusterPath, 'file')
        fprintf('using cluster_groups.csv  \n')
        [cids, cgs] = readClusterGroupsCSV(clusterPath);
    elseif exist(spikeClustersPath, 'file')
        fprintf('using spike_clusters.npy  \n')
        clu = readNPY(spikeClustersPath);
        cgs = 3*ones(size(unique(clu))); % all unsorted
    else
        fprintf('using spike_templates.npy  \n')
        clu = readNPY(spikeTemplatesPath);
        cgs = 3*ones(size(unique(clu))); % all unsorted
    end

    [cids, uQ, cR] = sqKilosort.maskedClusterQuality(resultsDirectory);

    isiV = sqKilosort.isiViolations(resultsDirectory) * 100; % in percent!
    
    histC = sqKilosort.histCompleteness(resultsDirectory);
