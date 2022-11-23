function process_TTcellbase(Directory)
    %{
    Directory: path to the cellbase directory
    %}

    listing = sort(getDir(Directory, 'file', 'TT'));	
    n_files = length(listing);

    TS_list = {};
    latest_spike = 0;
    for fn = 1:n_files
        try
            TS = load(fullfile(Directory, listing{fn})).TS; %.TS2;
        catch
            TS = load(fullfile(Directory, listing{fn})).TS1;
        end
     TS_list{fn} = TS;  
     latest_spike = max(max(TS), latest_spike); 
    end
    
    % spike times in ms
    latest_ms = uint64(round(latest_spike * 1000));
    spikes = zeros([n_files, latest_ms + 1], 'uint8');
    
    for i = 1:n_files
        spike_inds = uint64(round(TS_list{i} * 1000, 0)); 
        spikes(i, spike_inds) = 1;
    end
    
    save(fullfile(Directory, 'traces_ms.mat'), 'spikes', '-v7.3');

end  % process_TTcellbase()