function [inputs] = read_fidasim_inputs(fidasim_run_dir,run_id)
% This function reads the run_id_inputs.dat file created for FIDASIM.
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

% Get data:
% =========================================================================
file_path = prefix + "_inputs.dat";
inputs = read_fortran_namelist(file_path);

end