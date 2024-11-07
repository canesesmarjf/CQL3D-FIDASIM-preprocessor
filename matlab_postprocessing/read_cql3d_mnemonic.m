function [cql3d_nc] = read_cql3d_mnemonic(cql3d_run_dir,config)

% Ensure input data is correct:
% =========================================================================
% Ensure cql3d_run_dir ends with a separator
if ~endsWith(cql3d_run_dir, filesep)
    cql3d_run_dir = cql3d_run_dir + filesep;
end

% Remove any leading spaces or separators from plasma_file_name
plasma_file_name = regexprep(config.cql3d.plasma_file_name, '^[ /]*', ''); 

% Full file name:
file_name = cql3d_run_dir + plasma_file_name;    
    
% Read namelist:
% =========================================================================
info = ncinfo(file_name);

% Initialize an empty structure to store the fields
cql3d_nc = struct();

% Loop over each dataset in the netCDF file:
for i = 1:length(info.Variables)

    % Get the name of the dataset
    datasetName = info.Variables(i).Name;
    
    % Read the data from the netCDF file using the dataset name
    data = ncread(file_name, datasetName);

    % Handle character data
    if ischar(data)
        % If character data is more than 2D, flatten it
        if ndims(data) > 2
            % Skip this type of variable for now ...
            % data = join(string(reshape(data, [], size(data,3))), "", 1);
            data = [];
        elseif size(data, 1) > 1
            data = string(data');
        else
            data = string(data);
        end
    end
    
    % Dynamically assign the data to the plasma structure
    cql3d_nc.(datasetName) = data;
end
cql3d_nc.info = info;
% 
% 
% info = ncinfo(file_name);
%     powers = ncread(file_name,'/powers');
%     powers_int = ncread(file_name,'/powers_int'); % 'powers(*,6,k,t)=Ion particle source'
%     nbi_power_cql3d = powers_int(end,1,end); % [W]


end