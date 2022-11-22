function process_TTcellbase(Directory)
    file = fullfile(Directory, 'RecBehav.mat'); 
    load(file);
    
    
    list = sort(getDir(Directory, 'folder', 'cellbase')); 
    latest_spike = 0;
    
    for f = 1:length(list)
        cellbase = fullfile(Directory, list{f});
        listing = sort(getDir(cellbase, 'file', 'TT')); 

        TS_list = {};
        for fn = 1:length(listing)
            try
                TS = load(fullfile(cellbase, listing{fn})).TS2;
            catch
                TS = load(fullfile(cellbase, listing{fn})).TS1;
            end
         TS_list{fn} = TS;  
         latest_spike = max(max(TS), latest_spike) ; 
        end
        
        %spiketimes in ms
        spikes = zeros([length(listing), uint64(round(latest_spike*1000)) + 1], 'uint8');
        
        for i = 1:length(listing)
            spike_inds = uint64(round(TS_list{i}*1000, 0)); 
            spikes(i, spike_inds) = 1;
        end
        
        save(fullfile(cellbase, 'traces_ms.mat'), 'spikes', '-v7.3');
        
    end


end