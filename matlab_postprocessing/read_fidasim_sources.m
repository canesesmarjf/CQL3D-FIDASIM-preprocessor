function source = read_fidasim_sources(source_type,fidasim_run_dir, run_id)
% This function reads the HDF5 file that contains the sources produced by FIDASIM.
% The sources can be birth and sink points
% "source_type": "birth" or "sink"
% "run_id": name of the directory and run configuration
% "fidasim_run_dir": Parent directory that contains the "run_id" directory


% datasets: (/)
% dens   : Birth density: dens(beam_component,x,y,z) in beam grid coordinate XYZ
%         in units of fast-ions/(s*cm^3), 
%         size [3,nx,ny,nz]
% energy : Fast-ion birth energy: energy(particle) in [keV]. size [n_birth]
% ind    : Fast-ion birth beam grid indices: ind([i,j,k],particle), size [3,n_birth]
%         i =  ind(1,:), j = ind(2,:), k = ind(3,:)
% n_birth: Number of birth mc particles deposited
% pitch  : Fast-ion birth pitch w.r.t. the magnetic field: pitch(particle)
% ri     : Fast-ion birth position in R-Z-Phi: ri([r,z,phi],particle) in [cm,cm,radians]
% ri_gc  : Fast-ion guiding-center birth position in R-Z-Phi: ri_gc([r,z,phi],particle)
% type   : Fast-ion birth type (1=Full, 2=Half, 3=Third)
% vi     : Fast-ion birth velocity in R-Z-Phi: vi([r,z,phi],particle) in [cm/s]
% weight : Fast-ion birth weight: weight(particle) in [ions/s]

% Grid data: (/birth)
% nx    : Number of cells in the X direction
% ny    : Number of cells in the Y direction
% nz    : Number of cells in the Z direction
% x     : X value of cell center in beam grid coordinates (XYZ). size nx
% x_grid: X value of cell center in machine coordinates (UVW): x_grid(x,y,z),
%         size [nx,ny,nz]
% y     : Y value of cell center in beam grid coordinates (XYZ). size ny
% y_grid: Y value of cell center in machine coordinates (UVW): y_grid(x,y,z),
%         size [nx,ny,nz] 
% z     : Z value of cell center in beam grid coordinates (XYZ). size nz
% z_grid: Z value of cell center in machine coordinates (UVW): z_grid(x,y,z),
%         size [nx,ny,nz] 


% Ensure input data is correct:
% =========================================================================

% Standardize the input to lowercase and remove any trailing 's'
source_type = lower(source_type);
if endsWith(source_type, 's')
    source_type = extractBefore(source_type, strlength(source_type));
end

% Check for valid types
if strcmp(source_type, 'birth')
    disp('Source type is birth');
elseif strcmp(source_type,'birth_1')
    disp('Source type is birth_1');
elseif strcmp(source_type, 'sink')
    disp('Source type is sink');
else
    error('Invalid source_type. It must be "birth" or "sink" (any case, singular or plural).');
end
    
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
source_file_name = prefix + "_" + source_type + ".h5";

% Initialize an empty structure to store the fields
source = struct();

% Particle data:
if strcmpi(source_type,"birth") | strcmpi(source_type,"birth_1")
    source.n_birth = h5read(source_file_name,'/n_birth');
else
    source.n_sink = h5read(source_file_name,'/n_sink');
end
source.ri = h5read(source_file_name,'/ri'); % in r-z-phi cylindrical coordinates
source.vi = h5read(source_file_name,'/vi'); % vr-vz-vphi cyclindrical coords [cm/s]
source.weight = h5read(source_file_name,'/weight'); % [ions per sec]
source.energy = h5read(source_file_name,'/energy'); % [keV per ion]:

% Grid data:
% x,y,z coordinates in beam grid coordinates:
source.x = h5read(source_file_name,'/grid/x'); % size [nx]
source.y = h5read(source_file_name,'/grid/y'); % size [ny]
source.z = h5read(source_file_name,'/grid/z'); % size [nz]
% x,y,z grid coordinates in machine coordinates, size [nx,ny,nz]:
source.x_grid = h5read(source_file_name,'/grid/x_grid');
source.y_grid = h5read(source_file_name,'/grid/y_grid');
source.z_grid = h5read(source_file_name,'/grid/z_grid');
% source flux per unit volume profiles: [ions*cm^{-3}/sec]
% birth: size: [neut_type:3,nx,ny,nz]
% sink: size: [1,nx,ny,nz]
source.dens = h5read(source_file_name,'/dens');

% Derived quantities:
% =========================================================================

if strcmpi(source_type,"birth")
    % Birth flux per unit volume profiles:
    source.dens_full = permute(source.dens(1,:,:,:),[2 3 4 1]);
    source.dens_half = permute(source.dens(2,:,:,:),[2 3 4 1]);
    source.dens_third = permute(source.dens(3,:,:,:),[2 3 4 1]);
elseif strcmpi(source_type,"birth_1")
    % Birth_1 flux per unit volume profiles:
    source.dens_dcx = permute(source.dens(4,:,:,:),[2 3 4 1]);   
else
    % Rerrange data:
    sink.dens_1 = permute(source.dens(1,:,:,:),[2 3 4 1]);
end

% Grid cell volume:
source.dx = mean(diff(source.x));
source.dy = mean(diff(source.y));
source.dz = mean(diff(source.z));
source.dvol = source.dx*source.dy*source.dz; 

% Source position in cylindrical coords [r-z-phi]:
source.R = source.ri(1,:);
source.Z = source.ri(2,:);
source.phi = source.ri(3,:) + 0*pi/2;

% Convert to machine coordinates [UVW]:
source.U = source.R.*cos(source.phi);
source.V = source.R.*sin(source.phi);
source.W = source.Z;

% Source velocities [r-z-phi] cylindrical coords:
source.vR = source.vi(1,:);
source.vZ = source.vi(2,:);
source.vphi = source.vi(3,:); 

% Convert to cartesian:
source.vU = source.vR.*cos(source.phi) - source.vphi.*sin(source.phi);
source.vV = source.vR.*sin(source.phi) + source.vphi.*cos(source.phi);   
source.vW = source.vZ;

% Assuming that magnetic field is on the W direction:
source.vpar = source.vW;
source.vper = sqrt(source.vU.^2 + source.vV.^2);

% return
% 
% % Loop over each dataset in the HDF5 file
% for i = 1:length(info.Datasets)
%     % Get the name of the dataset
%     datasetName = info.Datasets(i).Name;
%     
%     % Read the data from the HDF5 file using the dataset name
%     data = h5read(source_file_name, ["/" + datasetName]);
%     
%     % Dynamically assign the data to the plasma structure
%     source.(datasetName) = data;
% end
% 
% info = h5info(source_file_name,'/grid');
% 
% % Loop over each dataset in the HDF5 file
% for i = 1:length(info.Datasets)
%     % Get the name of the dataset
%     datasetName = info.Datasets(i).Name;
%     
%     % Read the data from the HDF5 file using the dataset name
%     data = h5read(source_file_name, ["/grid/" + datasetName]);
%     
%     % Dynamically assign the data to the plasma structure
%     source.(datasetName) = data;
% end
% 
% 
% source.info = info;

% info = h5info(source_file_name,'/grid');



end