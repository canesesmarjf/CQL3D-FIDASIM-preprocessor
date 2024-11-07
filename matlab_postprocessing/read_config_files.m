function [config] = read_config_files(fidasim_run_dir, cql3d_run_dir, run_id)

% Ensure input data is correct:
% =========================================================================
% Ensure fidasim_run_dir ends with a separator
if ~endsWith(fidasim_run_dir, filesep)
    fidasim_run_dir = fidasim_run_dir + filesep;
end

% Ensure cql3d_run_dir ends with a separator
if ~endsWith(cql3d_run_dir, filesep)
    cql3d_run_dir = cql3d_run_dir + filesep;
end

% Remove any leading spaces or separators from run_id
run_id = regexprep(run_id, '^[ /]*', ''); 

% Assemble file prefix:
prefix_fida = fidasim_run_dir + run_id;
prefix_cql  = cql3d_run_dir + run_id;

% CQL3D:
file_path = prefix_cql + "_cql_config.nml";
config.cql3d = read_fortran_namelist(file_path);

% FIDASIM:
file_path = prefix_fida + "_config.nml";
config.fidasim = read_fortran_namelist(file_path);

end