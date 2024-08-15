% Plot inputs for WHAM FIDASIM thermal calculation:

clear all
close all
clc

read_birth = 1;
save_neutral_figs = 0;

scenario = 1;

switch scenario
    case 1
        runid = "WHAM_example";
        input_dir  = "./Step_1_output/";
        output_dir = "../Step_1_output/";       
end
%% Plasma:
% Get data from files:
plasma.dene = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/dene');
plasma.te = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/te');
plasma.ti = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/ti');
plasma.mask = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/mask');
plasma.r2d = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/r2d');
plasma.z2d = h5read(input_dir + runid + "_equilibrium.h5",'/plasma/z2d');

% Plot data:
figure('color','w')
Z = plasma.dene.*double(plasma.mask);
mesh(plasma.r2d, plasma.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('dene')

% Plot data:
figure('color','w')
Z = plasma.te.*double(plasma.mask);
mesh(plasma.r2d, plasma.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('te')

% Plot data:
figure('color','w')
Z = plasma.ti.*double(plasma.mask);
mesh(plasma.r2d, plasma.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('ti')

%% Fields:
% Get data from files:
info = h5info(input_dir + runid + "_equilibrium.h5",'/fields');
fields.bz = h5read(input_dir + runid + "_equilibrium.h5",'/fields/bz');
fields.bt = h5read(input_dir + runid + "_equilibrium.h5",'/fields/bt');
fields.br = h5read(input_dir + runid + "_equilibrium.h5",'/fields/br');
fields.mask = h5read(input_dir + runid + "_equilibrium.h5",'/fields/mask');
fields.r2d = h5read(input_dir + runid + "_equilibrium.h5",'/fields/r2d');
fields.z2d = h5read(input_dir + runid + "_equilibrium.h5",'/fields/z2d');

% Plot data:
figure('color','w')
Z = fields.bz.*double(fields.mask);
mesh(fields.r2d, fields.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('bz')

% Plot data:
figure('color','w')
Z = fields.bt.*double(fields.mask);
mesh(fields.r2d, fields.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('bt')

% Plot data:
figure('color','w')
Z = fields.br.*double(fields.mask);
mesh(fields.r2d, fields.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('br')


% Plot data:
figure('color','w')
Z = sqrt(fields.br.^2 + fields.bz.^2 + fields.bt.^2);
Z = Z.*double(fields.mask);
mesh(fields.r2d, fields.z2d, Z)
xlabel('R [cm]')
ylabel('Z [cm]')
title('bnorm')


%% Geometry:
info = h5info(input_dir + runid + "_geometry.h5",'/nbi');

uvw_src  = h5read(input_dir + runid + "_geometry.h5",'/nbi/src');
uvw_axis = h5read(input_dir + runid + "_geometry.h5",'/nbi/axis');
adist    = h5read(input_dir + runid + "_geometry.h5",'/nbi/adist');

%% input.dat
file_path = input_dir +  runid + "_inputs.dat";
input = read_namelist(file_path);

% Display the structure
disp(input);

% Extract data:
xmin = input.xmin;
xmax = input.xmax;
ymin = input.ymin;
ymax = input.ymax;
zmin = input.zmin;
zmax = input.zmax;
Pinj = input.pinj*1e6; % Injected NBI power in [W]

% Define the beam_grid.vertices of the rectangle in 3D space
beam_grid.vertices = [
    xmin, ymin, zmin;
    xmin, ymin, zmax;
    xmin, ymax, zmin;
    xmin, ymax, zmax;
    xmax, ymin, zmin;
    xmax, ymin, zmax;
    xmax, ymax, zmin;
    xmax, ymax, zmax
];

% Define the beam_grid.faces of the rectangle using the beam_grid.vertices
beam_grid.faces = [
    1, 2, 4, 3; % Side face (xmin)
    5, 6, 8, 7; % Side face (xmax)
    1, 2, 6, 5; % Side face (ymin)
    3, 4, 8, 7; % Side face (ymax)
    1, 3, 7, 5; % Bottom face
    2, 4, 8, 6  % Top face
];

% Beam grid orientation:
alpha = input.alpha;
beta  = input.beta;
ca = cos(alpha);
cb = cos(beta);
sa = sin(alpha);
sb = sin(beta);

beam_grid.rot_matrix = [ca*cb, -sa, ca*sb; ...
                        cb*sa, +ca, sa*sb; ...
                        -sb  ,  0 ,    cb];

beam_grid.origin = input.origin;

for ii = 1:size(beam_grid.vertices,1)
    uvw_vertices(ii,:) = beam_grid.origin + transpose(beam_grid.rot_matrix*transpose(beam_grid.vertices(ii,:)));
end

%% Get results from standard CQL3DM output:
file_name = "./Step_1_input/WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120.nc";
info = ncinfo(file_name);
powers = ncread(file_name,'/powers');
powers_int = ncread(file_name,'/powers_int'); % 'powers(*,6,k,t)=Ion particle source'
nbi_power_cql3d = powers_int(end,1,end); % [W]

%% Birth points:

if read_birth == 0
    return;
end

% Get data from files:
info = h5info(input_dir + runid + "_birth.h5");
n_birth = h5read(input_dir + runid + "_birth.h5",'/n_birth');
birth.ri = h5read(input_dir + runid + "_birth.h5",'/ri');

% Get grid data:
birth.x_grid = h5read(input_dir + runid + "_birth.h5",'/grid/x');
birth.y_grid = h5read(input_dir + runid + "_birth.h5",'/grid/y');
birth.z_grid = h5read(input_dir + runid + "_birth.h5",'/grid/z');

% Grid cell volume:
birth.dx_grid = mean(diff(birth.x_grid));
birth.dy_grid = mean(diff(birth.y_grid));
birth.dz_grid = mean(diff(birth.z_grid));
birth.dV_grid = birth.dx_grid*birth.dy_grid*birth.dz_grid; 

% Birth profiles: [ions*cm^{-3}/sec], size: [neut_type:3, nx:70, ny:50, nz:50]
birth.dens = h5read(input_dir + runid + "_birth.h5",'/dens');
birth.dens_full = permute(birth.dens(1,:,:,:),[2 3 4 1]);
birth.dens_half = permute(birth.dens(2,:,:,:),[2 3 4 1]);
birth.dens_third = permute(birth.dens(3,:,:,:),[2 3 4 1]);

% Birth position:
birth.R = birth.ri(1,:);
birth.Z = birth.ri(2,:);
birth.phi = birth.ri(3,:) + pi/2;

% Convert to cartesian:
birth.Y = birth.R.*cos(birth.phi);
birth.X = birth.R.*sin(birth.phi);

% Info on velocities "vi":
% 'Fast-ion birth velocity in R-Z-Phi: vi([r,z,phi],particle)'
% 'cm/s'

% Get weigths from birth points [ions per sec]:
birth.w = h5read(input_dir + runid + "_birth.h5",'/weight');

% Get energy of each birth particle in [keV per ion]:
birth.energy = h5read(input_dir + runid + "_birth.h5",'/energy');

% Total power in ions deposited in [W]:
Pabs = sum(birth.energy.*birth.w*1e3*e_c);
shinethrough = 1 - Pabs/Pinj;
disp(" ")
disp("(FIDASIM) Total NBI power injected: " + num2str(Pinj*1e-3,4) + " [kW]")
disp("(FIDASIM) Total NBI power absorbed: " + num2str(Pabs*1e-3,4) + " [kW]")
disp("(FIDASIM) " + num2str(shinethrough*100,4) + "% shine-through")

% CQL3D NBI power:
shinethrough = 1 - nbi_power_cql3d/Pinj;
disp(" ")
disp("(CQL3D-M) Total NBI power absorbed: " + num2str(nbi_power_cql3d*1e-3,4) + " [kW]")
disp("(CQL3D-M) " + num2str(shinethrough*100,4) + "% shine-through")

%% Plotting geometry:

% Compute the magnetic flux:
bflux = compute_magnetic_flux_2(fields);

% Obtain the magnetic flux at the plasma edge:
r_edge = 10/100; % [m]
bflux_LCFS = fields.bz(1,50)*pi*r_edge^2;

% Normalized magnetic flux
phi = bflux/bflux_LCFS;

% Define the contour level for the boundary between 0 and 1
contour_level = 1;

% Find the contours of the mask
figure()
contours = contourc(phi, [contour_level, contour_level]);

% Extract the coordinates of the boundary points
iz = round(contours(1, 2:end));
ir = round(contours(2, 2:end));

% Produce LCFS profile:
R = plasma.r2d(:,1);
Z = plasma.z2d(1,:)';
Z_lcfs = Z(iz);
R_lcfs = R(ir);

figure;
plot(R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
hold on;

%% Compute the LCFS contours:

% Compute the magnetic flux:
bflux = compute_magnetic_flux_2(fields);

% Obtain the magnetic flux at the plasma edge:
r_edge = 10/100; % [m]
numdim = size(fields.bz,2);
bflux_LCFS = fields.bz(1,round(numdim/2))*pi*r_edge^2;

% Normalized magnetic flux
phi = bflux/bflux_LCFS;

% Define the contour level for the boundary between 0 and 1
contour_level = 1;

% Find the contours of the mask
figure()
contours = contourc(phi, [contour_level, contour_level]);

% Extract the coordinates of the boundary points
iz = round(contours(1, 2:end));
ir = round(contours(2, 2:end));

% Produce LCFS profile:
Z_lcfs = Z(iz);
R_lcfs = R(ir);

% Plot the boundary points
if 1
figure;
plot(R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
hold on;
end

% Number of toroidal angles
numAngles = 30;

% Generate the toroidal angles
theta = linspace(0, 2*pi, numAngles);

% Preallocate arrays for the 3D coordinates
R3D = zeros(length(R_lcfs), numAngles);
Z3D = repmat(Z_lcfs, 1, numAngles);
X3D = zeros(length(R_lcfs), numAngles);
Y3D = zeros(length(R_lcfs), numAngles);

% Compute the coordinates for each toroidal angle
for i = 1:numAngles
    R3D(:, i) = R_lcfs;  % Radial coordinates are constant for each angle
    X3D(:, i) = R_lcfs * cos(theta(i));
    Y3D(:, i) = R_lcfs * sin(theta(i));
end

% Plot the 3D surface
font_size = 14;
figure('color','w');
plot3(X3D, Y3D, Z3D,'k-');
face_alpha = 0.2;
surf(X3D, Y3D, Z3D, 'EdgeColor', 'none','FaceColor',[0.3,0.75,0.93],'FaceAlpha',face_alpha)
set(gca,'FontSize',font_size)
xlabel('X [cm]','FontSize',font_size);
ylabel('Y [cm]','FontSize',font_size);
zlabel('Z [cm]','FontSize',font_size);
xlim([-70,70])
axis image
view([0,0])

% Calculate NBI path:
nbi_path = linspace(0,300);
uvw_nbi_path = uvw_src + uvw_axis*nbi_path;

% Aperture location:
uvw_aper = uvw_src + uvw_axis*adist;

% Plot NBI path:
hold on;
plot3(uvw_src(1),uvw_src(2),uvw_src(3),'go','MarkerSize',8)
plot3(uvw_aper(1),uvw_aper(2),uvw_aper(3),'ro','MarkerSize',8)
plot3(uvw_nbi_path(1,:),uvw_nbi_path(2,:),uvw_nbi_path(3,:),'k-','LineWidth',3)

% Plot the rectangle as a surface with transparency
hold on;
face_alpha = 0.2;
patch('vertices', uvw_vertices, 'faces', beam_grid.faces, 'FaceColor', 'blue',...
    'FaceAlpha', face_alpha, 'EdgeColor', 'none');
% Set axis properties
axis equal;
grid on;
view(3);

% Optionally, adjust the view and lighting for better visualization
camlight;
lighting gouraud;

% Add birth points:
plot3(birth.X,birth.Y,birth.Z,'k.')

% Add grid:
[xx,zz] = meshgrid(birth.x_grid,birth.z_grid);
plot3(zz,zeros(size(zz)),xx,'k')
plot3(zz',zeros(size(zz')),xx','k')

%% Neutral densities:
info = h5info(input_dir + runid + "_neutrals.h5");
disp({info.Groups.Name})
disp({info.Groups(2).Datasets.Name})
info.Groups(2).Datasets(1).Dataspace

denn_full_n = h5read(input_dir + runid + "_neutrals.h5",'/full/dens');
denn_half_n = h5read(input_dir + runid + "_neutrals.h5",'/half/dens');
denn_third_n = h5read(input_dir + runid + "_neutrals.h5",'/third/dens');
denn_dcx_n = h5read(input_dir + runid + "_neutrals.h5",'/dcx/dens');
denn_halo_n = h5read(input_dir + runid + "_neutrals.h5",'/halo/dens');

grid_x = h5read(input_dir + runid + "_neutrals.h5",'/grid/x_grid');
grid_y = h5read(input_dir + runid + "_neutrals.h5",'/grid/y_grid');
grid_z = h5read(input_dir + runid + "_neutrals.h5",'/grid/z_grid');

% These have dimensions of 70,50,50 (X,Y,Z)
denn_full = permute(sum(denn_full_n,1),[2,3,4,1]);
denn_half = permute(sum(denn_half_n,1),[2,3,4,1]);
denn_third = permute(sum(denn_third_n,1),[2,3,4,1]);
denn_dcx  = permute(sum(denn_dcx_n ,1),[2,3,4,1]);
denn_halo = permute(sum(denn_halo_n,1),[2,3,4,1]);

denn_full_XZ = permute(sum(denn_full,2),[1,3,2]);
denn_half_XZ = permute(sum(denn_half,2),[1,3,2]);
denn_third_XZ = permute(sum(denn_third,2),[1,3,2]);
denn_dcx_XZ  = permute(sum(denn_dcx ,2),[1,3,2]);
denn_halo_XZ = permute(sum(denn_halo,2),[1,3,2]);

caxis_min = 6.0;
caxis_max = 10.5;

figure('color','w')
hold on
logZ = log10(denn_full_XZ);
levels = 30;
line_style = 'none';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], full E')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_full_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_half_XZ);
levels = 30;
line_style = 'none';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], half E')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_half_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_third_XZ);
levels = 30;
line_style = 'none';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], third E')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_third_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_full_XZ + denn_half_XZ + denn_third_XZ);
levels = 30;
line_style = 'none';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], NBI')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_NBI_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_dcx_XZ);
levels = 15;
line_style = '-';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], DCX')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_dcx_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_halo_XZ);
levels = 12;
line_style = '-';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], Halo')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_halo_XZ"); 
end

figure('color','w')
hold on
logZ = log10(denn_dcx_XZ + denn_halo_XZ);
levels = 15;
line_style = '-';
contourf(permute(grid_x(:,1,:),[1,3,2]), permute(grid_z(:,1,:),[1,3,2]), logZ,levels,'linestyle',line_style)
plot(+R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
plot(-R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
title('log10 neutral density [cm^-2], DCX + Halo')
xlabel('R [cm]')
ylabel('Z [cm]')
axis image
ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_dhalo_XZ"); 
end

%% Distribution:

%% Functions:

function data = read_namelist(file_path)
    % Initialize an empty structure
    data = struct();

    % Open the file for reading
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Cannot open the file.');
    end

    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);

        % Skip comments and empty lines
        if isempty(line) || startsWith(line, '!!') || startsWith(line, '&') || startsWith(line, '/')
            continue;
        end

        % Parse the line
        [key, value] = parse_line(line);

        % Assign the value to the structure
        if ~isempty(key)
            % Handle array elements
            array_expr = regexp(key, '(.*)\((\d+)\)', 'tokens');
            if ~isempty(array_expr)
                base_key = array_expr{1}{1};
                index = str2double(array_expr{1}{2});
                if ~isfield(data, base_key)
                    data.(base_key) = [];
                end
                data.(base_key)(index) = value;
            else
                data.(key) = value;
            end
        end
    end

    % Close the file
    fclose(fid);
end

function [key, value] = parse_line(line)
    % Remove comments from the line
    line = regexprep(line, '!!.*', '');

    % Split the line into key and value
    parts = strsplit(line, '=');
    if length(parts) ~= 2
        key = '';
        value = '';
        return;
    end

    key = strtrim(parts{1});
    value_str = strtrim(parts{2});

    % Convert the value string to a MATLAB data type
    if startsWith(value_str, '''') && endsWith(value_str, '''')
        % String value
        value = strrep(value_str, '''', '');
    elseif contains(value_str, ',')
        % Array value
        value = str2num(value_str); %#ok<ST2NM>
    else
        % Single numeric value
        value = str2double(value_str);
        if isnan(value)
            value = value_str; % Keep as string if not a number
        end
    end
end

function save_fig(output_dir, file_name)

    figureName = [string(output_dir) + string(file_name)];

    % PDF figure:
    exportgraphics(gcf, figureName + ".pdf",'Resolution',600,'ContentType', 'vector') 

    % TIFF figure:
    exportgraphics(gcf, figureName + ".tiff",'Resolution',600) 

end

function phi = compute_magnetic_flux(fields)
    % Extract the magnetic field components and the grids
    Br = fields.br; % [T]
    Bz = fields.bz; % [T]
    r2d = fields.r2d*1e-2; % [m]
    z2d = fields.z2d*1e-2; % [m]
    
    % Get the size of the grid
    [nr, nz] = size(r2d);
    
    % Initialize the flux array
    phi = zeros(nr, nz);
    
    % Integrate Bz over the r direction
    for i = 2:nr
        phi(i, :) = phi(i-1, :) + Bz(i-1, :) .* (r2d(i, 1) - r2d(i-1, 1));
    end
    
    % Integrate Br over the z direction
    for j = 2:nz
        phi(:, j) = phi(:, j) + trapz(z2d(:, j), Br(:, j));
    end
end

function phi = compute_magnetic_flux_2(fields)
    % Extract the magnetic field components and the grids
    Br = fields.br;
    Bz = fields.bz;
    r2d = fields.r2d*1e-2; % [m]
    z2d = fields.z2d*1e-2; % [m]
    
    % Get the size of the grid
    [nr, nz] = size(r2d);
    
    % Initialize the flux array
    phi = zeros(nr, nz);
    
    % Integrate Bz over the r direction
    for i = 2:nr
        dr = r2d(i, 1) - r2d(i-1, 1);
        r_prime = (r2d(i, 1) + r2d(i-1, 1)) / 2; % Midpoint of r interval
        phi(i, :) = phi(i-1, :) + Bz(i-1, :) * 2 * pi * r_prime * dr;
    end
end


