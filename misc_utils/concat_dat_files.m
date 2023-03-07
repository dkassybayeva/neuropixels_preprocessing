function [ops] = concat_dat_files(ops, probe_dat_list, fproc, stub)
%{ 
    probe_dat_list: cell list of paths to the probe.dat files exported by 
    Trodes in the .rec/.kilosort subfolder

    fproc: path to merged output binary data FILE
    e.g., combined.mat

    edited by GK, March 6th, 2023
%}

% ----- open the output processed data file for writing ----- %
fidW = fopen(fproc, 'a+');  
if fidW<3
    error('Could not open %s for writing.', fproc);    
end
% ----------------------------------------------------------- %

NchanTOT = ops.NchanTOT;    % total number of channels in the raw binary file, including dead, auxiliary etc
NT = ops.NT;          % number of timepoints per batch
% NT = 40*(65600*2*NchanTOT); GK_note: I have no idea what this means
stub_bytes = NT*100*2*NchanTOT;

ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment



% file_lens = [];  % GK_note: doesn't seem to be used
tic;
for session_i = 1:length(probe_dat_list)
    fprintf('Time %3.0fs. Concatenating file... \n', toc);

    datafile = probe_dat_list{session_i};
    
    if stub
        %only making small snippet of file, should be 60 batches! 
        bytes = stub_bytes;
    else
        bytes = get_file_size(datafile);  % size in bytes of raw binary
    end
    
    % These ops params may change depending on the number of bytes
    nTimepoints = floor(bytes/NchanTOT/2);  % number of total timepoints
    ops.tend = min(nTimepoints, ceil(ops.trange(2) * ops.fs));  % ending timepoint
    ops.sampsToRead = ops.tend - ops.tstart;  % total number of samples to read

%     Nbatch = ceil(ops.sampsToRead /NT);  % GK_note: doesn't seem to be used

    %if i == 1
    %    ops.midpoint = Nbatch;
    %end
    
%     file_lens = [file_lens, bytes];  % GK_note: doesn't seem to be used

    %nTimepoints = nTimepoints + floor(bytes/NchanTOT/2); % number of total timepoints
    
    % ok we have to do something about this... 
    % would these parameters be per file?
    % 50 == reading larger chunks!
    % NT = 40*(65600*2*NchanTOT); % 2seconds * 30khz * 2bytes
    % Nbatch = ceil(bytes / 2*NchanTOT*NT);
    
    fid = fopen(datafile, 'r');  % open for reading raw data
    if fid<3
        error('Could not open %s for reading.', datafile);
    end
    
    c = 0;
    for offset = 0:NT:bytes
        fseek(fid, offset, 'bof');
        if ~mod(offset / NT, 1e4)
            disp(offset / bytes * 100)
        end

        
        %if we're at the end of the file only read the up to the end of the
        %file!
        if offset + NT >= bytes
            toread = bytes - offset;
        else
            toread = NT;
        end
        
        buff = fread(fid, toread / 2, '*int16');
        
        if isempty(buff)
            break; % this shouldn't really happen, unless we counted data batches wrong
        end
        

        count = fwrite(fidW, buff, 'int16'); % write this batch to binary file
        c = c+ count;
        if count~=numel(buff)
            error('Error writing batch %g to %s. Check available disk space.',offset,fproc);
        end
        
        
    end
    if ~(c == bytes / 2)
        error('Did not write the correct number of bytes.');
    end
    
    fclose(fid);
end

fclose(fidW);
