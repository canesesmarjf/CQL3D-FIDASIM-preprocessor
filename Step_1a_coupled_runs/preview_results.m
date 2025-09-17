% Script to preview results from time-dependent coupled runs

clear all
close all

%% Read sources text file:
file_name = "ion_source_points_FIDASIM.dat";
src_dir = "WHAM_Bob_IAEA_wall/";
fidasim_run_dir = "WHAM_Bob_IAEA_wall/";
cql3d_run_dir = "WHAM_Bob_IAEA_wall/";
run_id = "WHAM_Bob_IAEA_wall";
[data, headers, metadata] = read_source_data_FIDASIM(src_dir + file_name);

%% Read HDF5 source files:
% birth = read_fidasim_sources("birth",fidasim_run_dir, run_id);

%% Read mnemonic file:
config = read_config_files(fidasim_run_dir, cql3d_run_dir, run_id);
cql3d_nc = read_cql3d_mnemonic(cql3d_run_dir,config);

close all
tt = 6;

% Plot Ti profile:
figure;
mesh(cql3d_nc.solrz,cql3d_nc.solzz,squeeze(cql3d_nc.energyz(:,1,:,tt)))
zlim([0,50])
zlabel("[keV]")
title("Ion energy")

% Plot Te profile:
figure;
mesh(cql3d_nc.solrz,cql3d_nc.solzz,squeeze(cql3d_nc.energyz(:,2,:,tt)))
zlim([0,50])
zlabel("[keV]")
title("Electron energy")

% Plot ne profile:
figure;
mesh(cql3d_nc.solrz,cql3d_nc.solzz,squeeze(cql3d_nc.densz1(1,:,:,tt)))
zlim([0,1e15])
zlabel("[cm^{-3}]")
title("dene")

%% 

rng_birth = find(data(:,7) >= 0);
rng_sink = find(data(:,7) < 0);

%% Plot text file data:
figure('color','w')
s = scatter3(data(rng_birth,1), data(rng_birth,2), data(rng_birth,3), 10, 'r', 'filled');  % use scatter3
axis image

% Add the weight as custom data
s.UserData = data(:,7)*1e-14;

% Add custom tip
dt = s.DataTipTemplate;
dt.DataTipRows(end+1) = dataTipTextRow('Weight [1e14]', data(:,7)*1e-14);

%% Plot text file data:
figure('color','w')
s2 = scatter3(data(rng_sink,1), data(rng_sink,2), data(rng_sink,3), 10, 'g', 'filled');  % use scatter3
axis image

% Add the weight as custom data
s2.UserData = data(:,7)*1e-14;

% Add custom tip
dt = s2.DataTipTemplate;
dt.DataTipRows(end+1) = dataTipTextRow('Weight [1e14]', data(:,7)*1e-14);



%% Functions:
function [data, headers, metadata] = read_source_data_FIDASIM(filename)
%READ_SOURCE_DATA_FIDASIM Read particle data from a text file after <start-of-data> marker
%
%   [data, headers, metadata] = read_particle_data(filename)
%
%   Inputs:
%       filename - path to the text file
%
%   Outputs:
%       data     - Nx9 numeric array (x, y, z, vx, vy, vz, weight, denf4d, denf4d_per_marker)
%       headers  - cell array of column names
%       metadata - struct containing optional metadata (e.g., N)

    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    % Read line by line until <start-of-data>
    headers = {};
    metadata = struct();
    found_data_start = false;
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, '<start-of-data>')
            found_data_start = true;
            break;
        end

        % Extract metadata line: N =  ...
        if contains(line, 'N=') && contains(line, '#')
            tokens = regexp(line, 'N=\s*(\d+)', 'tokens');
            if ~isempty(tokens)
                metadata.N = str2double(tokens{1}{1});
            end
        end

        % Extract header line before <start-of-data>
        if contains(line, 'x(cm)')
            headers = strsplit(strtrim(line));
        end
    end

    if ~found_data_start
        fclose(fid);
        error('Data start marker "<start-of-data>" not found in file.');
    end

    % Read the remaining lines as numeric data
    data = [];
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            values = sscanf(line, '%f');
            if numel(values) == 9
                data(end+1, :) = values'; %#ok<AGROW>
            end
        end
    end

    fclose(fid);

    % Verify row count if N was found
    if isfield(metadata, 'N') && size(data, 1) ~= metadata.N
        warning('Expected %d data rows, but found %d.', metadata.N, size(data, 1));
    end
end