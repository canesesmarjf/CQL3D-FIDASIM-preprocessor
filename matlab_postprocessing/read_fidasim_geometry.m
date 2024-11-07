function [geom] = read_fidasim_geometry(fidasim_run_dir,run_id)
% This function reads the geomtry HDF5 files that are inputs to FIDASIM.
% "run_id": name of the directory and run configuration
% "fidasim_run_dir": Parent directory that contains the "run_id" directory

% Ensure input data is correct:
% =========================================================================
% Ensure fidasim_run_dir ends with a separator
if ~endsWith(fidasim_run_dir, filesep)
    fidasim_run_dir = fidasim_run_dir + filesep;
end

% Remove any leading spaces or separators from run_id
run_id = regexprep(run_id, '^[ /]*', ''); 

% Assemble file prefix:
prefix = fidasim_run_dir + run_id;

% Get plasma data:
% =========================================================================
info = h5info(prefix + "_geometry.h5",'/nbi');

% Initialize an empty structure to store the fields
geom = struct();

% Loop over each dataset in the HDF5 file
for i = 1:length(info.Datasets)
    % Get the name of the dataset
    datasetName = info.Datasets(i).Name;
    
    % Read the data from the HDF5 file using the dataset name
    data = h5read(prefix + "_geometry.h5", ['/nbi/' datasetName]);
    
    % Dynamically assign the data to the plasma structure
    geom.(datasetName) = data;
end
geom.info = info;

end