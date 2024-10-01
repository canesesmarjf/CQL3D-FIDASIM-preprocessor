% Plot inputs for WHAM FIDASIM thermal calculation:

clear all
close all
clc

read_birth = 1;
save_neutral_figs = 0;
calc_impact_vectors = 1;

scenario = 0;

switch scenario
    case 0
        runid = "WHAM_example";
        input_dir  = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_4_compute_FIDASIM_input_files/Step_1_output/";
        output_dir = input_dir;
        cqlinput_filepath = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_6_WHAM_FIDASIM_cases_Yuri/Step_1_input/";
        plasma_file_name = "WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120_yup5.nc";
        % cqlinput_filepath = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_6_WHAM_FIDASIM_cases_Yuri/Step_1_input/";
        % plasma_file_name = "WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120.nc";
    case 1
        runid = "ITER_neutral_wall_src";
        input_dir  = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_4_compute_FIDASIM_input_files/Step_1b_output/";
        output_dir = input_dir;
        cqlinput_filepath = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_4_compute_FIDASIM_input_files/Step_1b_input/";
        plasma_file_name = "none";
    case 99
        runid = "WHAM_example";
        input_dir  = "../run_archive/2024_09_05/fidasim_data/Step_1_output/";
        output_dir = input_dir;
        cqlinput_filepath = "../run_archive/2024_09_05/cql3d_data/";
        plasma_file_name = "WHAM2expander_NB100_nitr40npz2_iter0_sh02_iy300jx300lz120_i11.nc";        
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

% Machine coordiantes:
% W = Z_cyl
% ^
% |
% |
% X----->U

% NBI source in machine coordinates:
uvw_src  = h5read(input_dir + runid + "_geometry.h5",'/nbi/src');

% NBI direction vector in machine coordinats:
uvw_axis = h5read(input_dir + runid + "_geometry.h5",'/nbi/axis');

% Aperture distance along axis:
adist    = h5read(input_dir + runid + "_geometry.h5",'/nbi/adist');

%% input.dat
file_path = input_dir +  runid + "_inputs.dat";
input = read_namelist(file_path);

% Display the structure
disp(input);

% Beam grid dimensions in beam_grid coordinates:
xmin = input.xmin;
xmax = input.xmax;
ymin = input.ymin;
ymax = input.ymax;
zmin = input.zmin;
zmax = input.zmax;

% Injected NBI power:
Pinj = input.pinj*1e6; % Injected NBI power in [W]

% Define the beam_grid.vertices of the rectangle in 3D space
beam_grid.vertices = ...
[
    xmin, ymin, zmin;
    xmin, ymin, zmax;
    xmin, ymax, zmin;
    xmin, ymax, zmax;
    xmax, ymin, zmin;
    xmax, ymin, zmax;
    xmax, ymax, zmin;
    xmax, ymax, zmax
];

% Convert to a collection of 8 column vectors:
beam_grid.vertices = beam_grid.vertices';

% Define the beam_grid.faces of the rectangle using the beam_grid.vertices
beam_grid.faces = ...
[
    1, 2, 4, 3; % Side face (xmin)
    5, 6, 8, 7; % Side face (xmax)
    1, 2, 6, 5; % Side face (ymin)
    3, 4, 8, 7; % Side face (ymax)
    1, 3, 7, 5; % Bottom face
    2, 4, 8, 6  % Top face
];
beam_grid.faces = beam_grid.faces';

% Beam grid orientation:
alpha = input.alpha;
beta  = input.beta;
ca = cos(alpha);
cb = cos(beta);
sa = sin(alpha);
sb = sin(beta);

% Rotation matrix also referred to as basis in FIDASIM in subroutine
% tb_zxy:
% It is used to create the beam grid coordinate system [XYZ] by rotating the
% machine coorindate system [UVW] as follows:
% [XYZ] = [UVW]*basis

beam_grid.basis = [ca*cb, -sa, ca*sb; ...
                   cb*sa, +ca, sa*sb; ...
                   -sb  ,  0 ,    cb];

% Column vector for origin of beam grid in UVW:
beam_grid.origin_uvw = input.origin';

% Compute the beam grid vertices in UVW machine coordinate system:
for ii = 1:size(beam_grid.vertices,2)
    beam_grid.vertices_uvw(:,ii) = beam_grid.origin_uvw + beam_grid.basis*beam_grid.vertices(:,ii);
end

%% Get results from standard CQL3DM output:
if ~strcmpi(plasma_file_name,"none")
    file_name = cqlinput_filepath + plasma_file_name;    
    info = ncinfo(file_name);
    powers = ncread(file_name,'/powers');
    powers_int = ncread(file_name,'/powers_int'); % 'powers(*,6,k,t)=Ion particle source'
    nbi_power_cql3d = powers_int(end,1,end); % [W]
end

%% Ion birth points:

if read_birth == 0
    return;
end

% Get data from files:
info = h5info(input_dir + runid + "_birth.h5");
n_birth = h5read(input_dir + runid + "_birth.h5",'/n_birth');

% Ion birth positions in r-z-phi cylindrical coordinates:
birth.ri = h5read(input_dir + runid + "_birth.h5",'/ri');

% Ion birth velocity vectors in r-z-phi cyclindrical coords:
birth.vi = h5read(input_dir + runid + "_birth.h5",'/vi');

% Get grid data:
% 'X value of cell center in machine coordinates: x_grid(x,y,z)'
birth.U_grid = h5read(input_dir + runid + "_birth.h5",'/grid/x');
birth.V_grid = h5read(input_dir + runid + "_birth.h5",'/grid/y');
birth.W_grid = h5read(input_dir + runid + "_birth.h5",'/grid/z');

% Grid cell volume:
birth.dx_grid = mean(diff(birth.U_grid));
birth.dy_grid = mean(diff(birth.V_grid));
birth.dz_grid = mean(diff(birth.W_grid));
birth.dV_grid = birth.dx_grid*birth.dy_grid*birth.dz_grid; 

% Birth profiles: [ions*cm^{-3}/sec], size: [neut_type:3, nx:70, ny:50, nz:50]
birth.dens = h5read(input_dir + runid + "_birth.h5",'/dens');
birth.dens_full = permute(birth.dens(1,:,:,:),[2 3 4 1]);
birth.dens_half = permute(birth.dens(2,:,:,:),[2 3 4 1]);
birth.dens_third = permute(birth.dens(3,:,:,:),[2 3 4 1]);

% Birth position in cylindrical coords [r-z-phi]:
birth.R = birth.ri(1,:);
birth.Z = birth.ri(2,:);
birth.phi = birth.ri(3,:) + 0*pi/2;

% Convert to machine coordinates [UVW]:
birth.U = birth.R.*cos(birth.phi);
birth.V = birth.R.*sin(birth.phi);
birth.W = birth.Z;

% Ion birth velocities [r-z-phi] cylindrical coords:
birth.vR = birth.vi(1,:);
birth.vZ = birth.vi(2,:);
birth.vphi = birth.vi(3,:); 

% Convert to cartesian:
birth.vU = birth.vR.*cos(birth.phi) - birth.vphi.*sin(birth.phi);
birth.vV = birth.vR.*sin(birth.phi) + birth.vphi.*cos(birth.phi);   
birth.vW = birth.vZ;

% Assuming that magnetic field is on the W direction:
birth.vpar = birth.vW;
birth.vper = sqrt(birth.vU.^2 + birth.vV.^2);

% Info on velocities "vi":
% 'Fast-ion birth velocity in R-Z-Phi: vi([r,z,phi],particle)'
% 'cm/s'

% Get weigths from birth points [ions per sec]:
birth.weight = h5read(input_dir + runid + "_birth.h5",'/weight');

% Get energy of each birth particle in [keV per ion]:
birth.energy = h5read(input_dir + runid + "_birth.h5",'/energy');

[birth_f, birth_vpar, birth_vper] = compute_weighted_velocity_distribution(birth.vpar, birth.vper, birth.weight, 20, 20);

% Total power in ions deposited in [W]:
Pabs = sum(birth.energy.*birth.weight*1e3*e_c);
shinethrough = 1 - Pabs/Pinj;
disp(" ")
disp("(FIDASIM) Total NBI power injected: " + num2str(Pinj*1e-3,4) + " [kW]")
disp("(FIDASIM) Total NBI power absorbed: " + num2str(Pabs*1e-3,4) + " [kW]")
disp("(FIDASIM) " + num2str(shinethrough*100,4) + "% shine-through")

% CQL3D NBI power:
if ~strcmpi(plasma_file_name,"none")
    shinethrough = 1 - nbi_power_cql3d/Pinj;
    disp(" ")
    disp("(CQL3D-M) Total NBI power absorbed: " + num2str(nbi_power_cql3d*1e-3,4) + " [kW]")
    disp("(CQL3D-M) " + num2str(shinethrough*100,4) + "% shine-through")
end

if calc_impact_vectors
% Dimensions of the cylindrical surface:
% =========================================================================
geom.R = 50;   % Radius of the cylinder in [cm]
geom.L = 200;   % Length of the cylinder in [cm]
geom.z0 = 0.0;  % z-coordinate of the center of the cylinder
geom.axis_cyl = [0,0,1];

% Compute the impact points in cartesian coordinates:
% =========================================================================
p_vec = zeros(n_birth,3);
v_vec = zeros(n_birth,3);
impact_vector = zeros(n_birth,3);
impact_surface = zeros(n_birth,3);

for ii = 1:n_birth
    p_vec(ii,:) = [birth.U(ii),birth.V(ii),birth.W(ii)];
    v_vec(ii,:) = [birth.vU(ii),birth.vV(ii),birth.vW(ii)];
end
[impact_vector,impact_surface] = calculate_impact_vector_on_cylinder(p_vec,v_vec,geom);

% Convert impact vectors to cylindrical coordinates:
% =========================================================================
impact_vector_rphiz = convert_impact_data_to_cyl_coords(impact_vector, impact_surface, geom);

% Calculate heat map:
num_z_bins = 100;
num_theta_bins = 81;
ii_rng = find(impact_surface == 1);
[theta_bins, z_bins, heat_map] = accumulate_impact_vector(impact_vector_rphiz(ii_rng,:), geom, birth.weight(ii_rng), num_theta_bins, num_z_bins);

% Plot the heat map:
% =========================================================================
plot_heat_map_2d(theta_bins, z_bins, heat_map)
end

 %% Ion sink points:

% Check if file exits:
file_name = input_dir + runid + "_sink.h5";
if exist(file_name, 'file') == 2
    disp('Ion file exists');
    
    % Get data from files:
    info = h5info(file_name);
    n_sink = h5read(file_name,'/n_sink');

    % Ion birth positions in r-z-phi cylindrical coordinates:
    sink.ri = h5read(file_name,'/ri');    

    % Ion brith velocity vectors in r-z-phi cyclindrical coords:
    sink.vi = h5read(file_name,'/vi');

    % Sink profiles: [ions*cm^{-3}/sec], size: [thermal_mass:1, nx:70, ny:50, nz:50]
    sink.dens = h5read(input_dir + runid + "_sink.h5",'/dens');
    sink.dens_1 = permute(sink.dens(1,:,:,:),[2 3 4 1]);

    % Sink position in cylindrical coords::
    sink.R = sink.ri(1,:);
    sink.Z = sink.ri(2,:);
    sink.phi = sink.ri(3,:) + 0*pi/2;
    
    % Convert to machine coordinates [UVW]:
    sink.U = sink.R.*cos(sink.phi);
    sink.V = sink.R.*sin(sink.phi);
    sink.W = sink.Z;

    % Sink velocities [r-z-phi] in cylindrical:
    sink.vR = sink.vi(1,:);
    sink.vZ = sink.vi(2,:);
    sink.vphi = sink.vi(3,:); 

    % Convert to machine coordinates [UVW]:
    sink.vU = sink.vR.*cos(sink.phi) - sink.vphi.*sin(sink.phi);
    sink.vV = sink.vR.*sin(sink.phi) + sink.vphi.*cos(sink.phi);   
    sink.vW = sink.vZ;

    % Assuming that magnetic field is on the W direction:
    sink.vpar = sink.vW;
    sink.vper = sqrt(sink.vU.^2 + sink.vV.^2);

    % Info on velocities "vi":
    % 'Ion sink velocity in R-Z-Phi: vi([r,z,phi],particle)'
    % 'cm/s'
    
    % Get weigths from sink points [ions per sec]:
    sink.weight = h5read(file_name,'/weight');
    
    % Get energy of each sink particle in [keV per ion]:
    sink.energy = h5read(file_name,'/energy');
    npar = 70;
    nper = 35;
    [sink_f, sink_vpar, sink_vper] = compute_weighted_velocity_distribution(sink.vpar, sink.vper, sink.weight, npar, nper);
    
    % Plot ion sink profile:
    figure('color','w')
    box on
    hold on
    surf(sink_vpar, sink_vper, sink_f','LineStyle','none');
    surf(birth_vpar, birth_vper, 0.05*birth_f','LineStyle','none');

    % Beam energies:
    [e_levels,~,~] = unique(birth.energy);
    ie = 1:10:numel(birth.energy);
    hnbi = plot3(birth.vpar(ie), birth.vper(ie), max(sink_f(:))*ones(size(ie)),'go');
    hnbi.MarkerEdgeColor = 'g';
    hnbi.MarkerFaceColor = hnbi.MarkerEdgeColor;

    set(gca, 'YDir', 'normal');
    xlabel('$v_{\parallel}$ [cm/s]','Interpreter','latex','FontSize',18);
    ylabel('$v_{\perp}$ [cm/s]','Interpreter','latex','FontSize',18);
    title({'Ion sink profile';'$f_+(\vec{v})\int{ f_B(\vec{v}'')\langle{\sigma v_{rel}}\rangle_\phi^{cx} d^3 v''}$'},'Interpreter','latex','FontSize',16); % (\vec{v},\vec{v}'')
    set(gca,'ZScale','lin')
    colormap(flipud(hot))
    caxis([0,max(max(sink_f))])
    grid on;
    set(gca,'PlotBoxAspectRatio',[1 0.5 1])
    colorbar;

    % Draw a circle at various energies:
    energy_vec = [40,60,80,100]*1e3;
    vel_vec = 1e2*sqrt(2*e_c*energy_vec/(2*m_p)); % [cm/s]
    theta = linspace(0,2*pi,1e2);
    line_color = {'g','m','c','k'};
    hold on; 
    for ii = 1:numel(vel_vec)
        hline(ii)=plot3(vel_vec(ii)*cos(theta),vel_vec(ii)*sin(theta),5e24*ones(size(theta)),line_color{ii},'LineWidth',2);
        legend_text{ii} = num2str(energy_vec(ii)*1e-3,3) + " [keV]";
    end
    legend_text{end+1} = "NBI";
    legend([hline,hnbi],legend_text)

    %     vmax = max(sink_vper);
    vmax = vel_vec(3);
    xlim([-vmax,vmax]*1.2)
    ylim([    0,vmax]*1.2)

    % Total power in ions deposited in [W]:
    Pcx = sum(sink.energy.*sink.weight*1e3*e_c);
    disp(" ")
    disp("(FIDASIM) Total power lost to CX: " + num2str(Pcx*1e-3,4) + " [kW]")

    % Ratio between sink current and birth current:
    r_cx = sum(sink.weight) / sum(birth.weight);
    disp(" ")
    disp("(FIDASIM) r_cx = sum(sink.w)/sum(birth.w): " + num2str(r_cx,3))

    if calc_impact_vectors
    % Dimensions of the cylindrical surface:
    % =========================================================================
    geom.R = 50;   % Radius of the cylinder in [cm]
    geom.L = 200;   % Length of the cylinder in [cm]
    geom.z0 = 0.0;  % z-coordinate of the center of the cylinder
    geom.axis_cyl = [0,0,1];

    % Compute the impact points in cartesian coordinates:
    % =========================================================================
    p_vec = zeros(n_sink,3);
    v_vec = zeros(n_sink,3);
    impact_vector = zeros(n_sink,3);
    impact_surface = zeros(n_sink,3);

    for ii = 1:n_sink
        p_vec(ii,:) = [sink.U(ii),sink.V(ii),sink.W(ii)];
        v_vec(ii,:) = [sink.vU(ii),sink.vV(ii),sink.vW(ii)];
    end
    [impact_vector,impact_surface] = calculate_impact_vector_on_cylinder(p_vec,v_vec,geom);
    
    % Convert impact vectors to cylindrical coordinates:
    % =========================================================================
    impact_vector_rphiz = convert_impact_data_to_cyl_coords(impact_vector, impact_surface, geom);
    
    % Calculate CX impact map on vessel:
    % =========================================================================
    num_z_bins = 100;
    num_theta_bins = 81;
    ii_rng = find(impact_surface == 1);
    [theta_bins, z_bins, cx_neutral_flux_wall] = accumulate_impact_vector(impact_vector_rphiz(ii_rng,:), geom, sink.weight(ii_rng), num_theta_bins, num_z_bins);
    dz = mean(diff(z_bins)); % [cm]
    ds = geom.R*mean(diff(theta_bins)); % [cm]
    cx_neutral_flux_wall = cx_neutral_flux_wall/(dz*ds); % [ion-cm^{-2}/s]

    % Calculate power deposition on vaccum vessel:
    % =========================================================================
    marker_weight =  sink.weight(ii_rng).* sink.energy(ii_rng)*(1e3)*e_c; % [W/ion]
    [theta_bins, z_bins, cx_energy_flux_wall] = accumulate_impact_vector(impact_vector_rphiz(ii_rng,:), geom, marker_weight, num_theta_bins, num_z_bins);
    cx_energy_flux_wall = cx_energy_flux_wall/(dz*ds); % [W-cm^{-2}]

    % Plot the heat map:
    % =========================================================================
    plot_heat_map_2d(theta_bins, z_bins, cx_neutral_flux_wall)
    title('CX neutral flux density on vaccum vessel [atom/cm^2/s]');

    % Plot power map:
    % =========================================================================
    plot_heat_map_2d(theta_bins, z_bins, cx_energy_flux_wall)
    title('CX energy flux density on vaccum vessel [W/cm^2]');
    end
end

%% Plotting geometry:

% Compute the magnetic flux:
bflux = compute_magnetic_flux_2(fields);

% Obtain edge of plasma:
R = plasma.r2d(:,1);
ii = round(size(plasma.dene,1)/2);
ne_R = plasma.dene(:,ii);
ii_edge = find(ne_R > 1e12);
r_edge = R(ii_edge(end));

% Magnetic flux at the edge:
bflux_LCFS = bflux(ii_edge(end),ii); 

% Obtain the magnetic flux at the plasma edge:
% r_edge = 10/100; % [m]
% bflux_LCFS = fields.bz(1,50)*pi*r_edge^2;

% Normalized magnetic flux
phi = bflux/bflux_LCFS;

% Define the contour level for the boundary between 0 and 1
contour_level = 1;

% Find the contours of the mask
contours = contourc(phi, [contour_level, contour_level]);

% Extract the coordinates of the boundary points
iz = round(contours(1, 2:end));
ir = round(contours(2, 2:end));

% Produce LCFS profile:
R = plasma.r2d(:,1);
Z = plasma.z2d(1,:)';
Z_lcfs = Z(iz);
R_lcfs = R(ir);

if 1
figure;
plot(R_lcfs,Z_lcfs, 'r-', 'LineWidth', 2);
hold on;
end

%% Compute the LCFS contours:

% % Compute the magnetic flux:
% bflux = compute_magnetic_flux_2(fields);
% 
% % Obtain the magnetic flux at the plasma edge:
% r_edge = 10/100; % [m]
% numdim = size(fields.bz,2);
% bflux_LCFS = fields.bz(1,round(numdim/2))*pi*r_edge^2;

% Normalized magnetic flux
phi = bflux/bflux_LCFS;

% Define the contour level for the boundary between 0 and 1
contour_level = 1;

% Find the contours of the mask
contours = contourc(phi, [contour_level, contour_level]);

% Extract the coordinates of the boundary points
iz = round(contours(1, 2:end));
ir = round(contours(2, 2:end));

% Produce LCFS profile:
Z_lcfs = Z(iz);
R_lcfs = R(ir);

% Plot the boundary points
if 0
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
U3D = zeros(length(R_lcfs), numAngles);
V3D = zeros(length(R_lcfs), numAngles);
W3D = Z3D;

% Compute the coordinates for each toroidal angle
for i = 1:numAngles
    R3D(:, i) = R_lcfs;  % Radial coordinates are constant for each angle
    U3D(:, i) = R_lcfs * cos(theta(i));
    V3D(:, i) = R_lcfs * sin(theta(i));
end

% Plot the 3D surface
font_size = 14;
h3d = figure('color','w');
plot3(U3D, V3D, W3D,'k-');
face_alpha = 0.2;
surf(U3D, V3D, W3D, 'EdgeColor', 'none','FaceColor',[0.3,0.75,0.93],'FaceAlpha',face_alpha)
set(gca,'FontSize',font_size)
xlabel('X [cm]','FontSize',font_size);
ylabel('Y [cm]','FontSize',font_size);
zlabel('Z [cm]','FontSize',font_size);
xlim([-70,70]*2)
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
patch('vertices', beam_grid.vertices_uvw', 'faces', beam_grid.faces', 'FaceColor', 'blue',...
    'FaceAlpha', face_alpha, 'EdgeColor', 'none');
% Set axis properties
axis equal;
grid on;
view(3);

% Optionally, adjust the view and lighting for better visualization
camlight;
lighting gouraud;

% Add birth points:
plot3(birth.U,birth.V,birth.W,'k.')

% Add birth quiver:
if 0
vmax = max(sqrt(birth.vU.^2 + birth.vV.^2 + birth.vW.^2));
sf = 50/vmax;
rng = randi(n_birth,[1e3,1]);
U = birth.U(rng);
V = birth.V(rng);
W = birth.W(rng);
vU = birth.vU(rng);
vV = birth.vV(rng);
vW = birth.vW(rng);
hq(1) = quiver3(U,V,W,sf*vU,sf*vV,sf*vW,0);
set(hq(1),'MaxHeadSize',0.01,'color','k','LineWidth',0.1,'LineStyle','--')
end

% Add sink points:
if 1
plot3(sink.U,sink.V,sink.W,'g.')
end

% Add sink quiver:
if 0
vmax = max(sqrt(sink.vU.^2 + sink.vV.^2 + sink.vW.^2));
sf = 50/vmax;
rng = randi(n_sink,[5e3,1]);
U = sink.U(rng);
V = sink.V(rng);
W = sink.W(rng);
vU = sink.vU(rng);
vV = sink.vV(rng);
vW = sink.vW(rng);
hq(2) = quiver3(U,V,W,sf*vU,sf*vV,sf*vW,0);
set(hq(2),'MaxHeadSize',0.01,'color','g','LineWidth',0.1,'LineStyle','-')
end

% Add cx power density impacting vaccum vessel:
if calc_impact_vectors
    flux_wall = cx_energy_flux_wall;
    plot_heat_map_3d(h3d, theta_bins, z_bins, flux_wall, geom)
    caxis([0,0.8*max(flux_wall(:))])
    lighting none
end

% Add grid:
if 0
    [UU,WW] = meshgrid(birth.U_grid,birth.W_grid);
    plot3(WW,zeros(size(WW)),UU,'k')
    plot3(WW',zeros(size(WW')),UU','k')
end

%% Neutral densities:
info = h5info(input_dir + runid + "_neutrals.h5");
disp(" ")
disp(" Neutral densities: ")

% Diagnostics:
if 0
    disp({info.Groups.Name})
    disp({info.Groups(2).Datasets.Name})
    disp(info.Groups(2).Datasets(1).Dataspace)
end

denn_full_n = h5read(input_dir + runid + "_neutrals.h5",'/full/dens');
denn_half_n = h5read(input_dir + runid + "_neutrals.h5",'/half/dens');
denn_third_n = h5read(input_dir + runid + "_neutrals.h5",'/third/dens');

grid_x = h5read(input_dir + runid + "_neutrals.h5",'/grid/x_grid');
grid_y = h5read(input_dir + runid + "_neutrals.h5",'/grid/y_grid');
grid_z = h5read(input_dir + runid + "_neutrals.h5",'/grid/z_grid');

% These have dimensions of 70,50,50 (X,Y,Z)
denn_full = permute(sum(denn_full_n,1),[2,3,4,1]);
denn_half = permute(sum(denn_half_n,1),[2,3,4,1]);
denn_third = permute(sum(denn_third_n,1),[2,3,4,1]);

denn_full_XZ = permute(sum(denn_full,2),[1,3,2]);
denn_half_XZ = permute(sum(denn_half,2),[1,3,2]);
denn_third_XZ = permute(sum(denn_third,2),[1,3,2]);

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
% ylim([-80,80])
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
% ylim([-80,80])
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
% ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_third_XZ"); 
end

%% Halo and DCX neutral densities:

try
    denn_dcx_n = h5read(input_dir + runid + "_neutrals.h5",'/dcx/dens');
    denn_halo_n = h5read(input_dir + runid + "_neutrals.h5",'/halo/dens');
    denn_dcx  = permute(sum(denn_dcx_n ,1),[2,3,4,1]);
    denn_halo = permute(sum(denn_halo_n,1),[2,3,4,1]);
    denn_dcx_XZ  = permute(sum(denn_dcx ,2),[1,3,2]);
    denn_halo_XZ = permute(sum(denn_halo,2),[1,3,2]);
catch
    disp("DCX and halo densities not available ...")
    return
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
% ylim([-80,80])
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
% ylim([-80,80])
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
% ylim([-80,80])
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
% ylim([-80,80])
caxis([caxis_min,caxis_max])
colorbar

if save_neutral_figs
    save_fig(output_dir,"neutral_density_dhalo_XZ"); 
end

%% Distribution:

%% Beam grid plotting:


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
    %line = regexprep(line, '!!.*', '');

    % Remove comments that start with single '!' or '!!' anywhere in the line
    line = regexprep(line, '^\s*!.*|!!.*', '');

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

function [N_weighted, Xcenters, Ycenters] = compute_weighted_velocity_distribution(V1, V2, W, nbinsX, nbinsY)
    % compute_weighted_velocity_distribution - Compute the weighted distribution
    % of velocities on a 2D grid.
    %
    % Syntax:  [N_weighted, Xcenters, Ycenters] = compute_weighted_velocity_distribution(V1, V2, W, nbinsX, nbinsY)
    %
    % Inputs:
    %   V1 - 1D array of parallel velocities
    %   V2 - 1D array of perpendicular velocities
    %   W  - 1D array of weights corresponding to V1 and V2
    %   nbinsX - Number of bins for V1 (x-axis)
    %   nbinsY - Number of bins for V2 (y-axis)
    %
    % Outputs:
    %   N_weighted - 2D array containing the weighted distribution
    %   Xcenters - Centers of the bins for V1
    %   Ycenters - Centers of the bins for V2
    
    % Determine the range of V1 and V2 with an extra 20% space
    rangeV1 = max(V1) - min(V1);
    rangeV2 = max(V2) - min(V2);
    
    minV1 = min(V1) - 0.1 * rangeV1;
    maxV1 = max(V1) + 0.1 * rangeV1;
    minV2 = min(V2) - 0.1 * rangeV2;
    maxV2 = max(V2) + 0.1 * rangeV2;
    
    % Create a uniform grid spanning the adjusted range of V1 and V2
    Xedges = linspace(minV1, maxV1, nbinsX + 1);
    Yedges = linspace(minV2, maxV2, nbinsY + 1);
    
    % Calculate the centers of each bin
    Xcenters = (Xedges(1:end-1) + Xedges(2:end)) / 2;
    Ycenters = (Yedges(1:end-1) + Yedges(2:end)) / 2;
    
    % Initialize the weighted distribution grid
    N_weighted = zeros(nbinsX, nbinsY);
    
    % Loop over each particle and accumulate the weight in the appropriate cell
    for i = 1:length(V1)
        % Find the indices of the bin that V1(i) and V2(i) fall into
        [~, binX] = histc(V1(i), Xedges);
        [~, binY] = histc(V2(i), Yedges);
        
        % Ensure that the bin indices are within valid ranges
        if binX > 0 && binX <= nbinsX && binY > 0 && binY <= nbinsY
            N_weighted(binX, binY) = N_weighted(binX, binY) + W(i)*V2(i);
        end
    end
end

function [sink_v_dist, Vcenters] = compute_1D_velocity_distribution(sink_f, sink_vpar, sink_vper)
    % compute_1D_velocity_distribution - Compute the 1D velocity distribution
    % from a 2D distribution of parallel and perpendicular velocities.
    %
    % Syntax: [sink_v_dist, Vcenters] = compute_1D_velocity_distribution(sink_f, sink_vpar, sink_vper)
    %
    % Inputs:
    %   sink_f    - 2D array of weighted distribution values (e.g., sink_f)
    %   sink_vpar - Centers of the bins for parallel velocities
    %   sink_vper - Centers of the bins for perpendicular velocities
    %
    % Outputs:
    %   sink_v_dist - 1D array containing the weighted distribution of the magnitude of velocity
    %   Vcenters    - Centers of the bins for the velocity magnitude

    % Create a 2D meshgrid for vpar and vper
    [VparGrid, VperGrid] = meshgrid(sink_vpar, sink_vper);

    % Calculate the magnitude of the velocity for each bin
    sink_v = sqrt(VparGrid.^2 + VperGrid.^2);

    % Determine the range and bin edges for the velocity magnitude
    Vmin = min(sink_v(:));
    Vmax = max(sink_v(:));
    nVbins = length(sink_vpar);  % Number of bins for 1D distribution (you can adjust this)
    Vedges = linspace(Vmin, Vmax, nVbins + 1);

    % Initialize the 1D distribution array
    sink_v_dist = zeros(1, nVbins);

    % Loop over each bin in the 2D distribution
    for i = 1:numel(sink_f)
        % Find the appropriate bin for the current velocity magnitude
        [~, binV] = histc(sink_v(i), Vedges);

        % Ensure the bin index is within valid range
        if binV > 0 && binV <= nVbins
            % Accumulate the weight from the 2D distribution into the 1D distribution
            sink_v_dist(binV) = sink_v_dist(binV) + sink_f(i);
        end
    end

    % Calculate the centers of the bins for the 1D distribution
    Vcenters = (Vedges(1:end-1) + Vedges(2:end)) / 2;
end

function namelist = readFortranNamelist(filename)
    % Open the file
    fid = fopen(filename, 'r');
    
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    namelist = struct();
    groupName = '';
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        
        if isempty(line) || startsWith(line, '!')
            % Skip empty lines or comments
            continue;
        end
        
        % Check for the start of a new namelist group
        if startsWith(line, '&')
            groupName = line(2:end);
            namelist.(groupName) = struct();
        elseif startsWith(line, '/')
            % End of namelist group
            groupName = '';
        elseif ~isempty(groupName)
            % Parse the variables in the group
            [varName, varValue] = strtok(line, '=');
            varName = strtrim(varName);
            varValue = strtrim(varValue(2:end)); % Remove '='
            
            % Handle different types of variables
            if contains(varValue, '"') || contains(varValue, '''')
                % String value
                varValue = strrep(varValue, '"', '');
                varValue = strrep(varValue, '''', '');
            elseif contains(varValue, '.')
                % Floating point number
                varValue = str2double(varValue);
            else
                % Integer
                varValue = str2double(varValue);
            end
            
            namelist.(groupName).(varName) = varValue;
        end
    end
    
    % Close the file
    fclose(fid);
end

function [impact_vector, impact_surface] = calculate_impact_vector_on_cylinder(p, v, geom)
    % p: Matrix of starting points [N x 3]
    % v: Matrix of velocity vectors [N x 3]
    % geom: Structure containing the cylinder geometry with fields:
    %       - R: Radius of the cylinder
    %       - L: Length of the cylinder along the z-axis
    %       - z0: z-coordinate of the center of the cylinder
    %       - axis_cyl: Unit vector representing the direction of the cylinder's axis
    % impact_vector: Matrix of intersection points [M x 3]
    % impact_surface: Array of integers indicating which surface was intersected:
    %                 1 for cylindrical, 2 for top cap, 3 for bottom cap

    % Normalize the axis vector
    axis_cyl = geom.axis_cyl;
    axis_cyl = axis_cyl / norm(axis_cyl);
    
    % Define the rotation matrix to align the cylinder axis with the z-axis
    z_axis = [0, 0, 1];
    rotation_matrix = get_rotation_matrix(axis_cyl, z_axis);
    inverse_rotation_matrix = rotation_matrix'; % Transpose of the rotation matrix

    % Extract the cylinder parameters from the geom structure
    R = geom.R;
    L = geom.L;
    z_center = geom.z0;

    % Calculate the z-coordinate of the bottom and top bases
    z_bottom = z_center - L/2;
    z_top = z_center + L/2;

    % Estimate the maximum number of potential intersections
    max_intersections = 2 * size(p, 1);  % Two intersections per vector as an estimate

    % Preallocate arrays
    impact_vector = zeros(max_intersections, 3);
    impact_surface = zeros(max_intersections, 1);

    current_index = 0;

    % Loop through each pair of p and v
    for ii = 1:size(p, 1)
        % Rotate the points and vectors to align with the z-axis
        p_rotated = (rotation_matrix * p(ii, :)')';
        v_rotated = (rotation_matrix * v(ii, :)')';

        % Unpack the rotated vector components
        x0 = p_rotated(1);
        y0 = p_rotated(2);
        z0_current = p_rotated(3); % This is the z-coordinate of the starting point
        
        vx = v_rotated(1);
        vy = v_rotated(2);
        vz = v_rotated(3);

        % 1. Check intersection with the curved cylindrical surface
        A = vx^2 + vy^2;
        B = 2 * (x0 * vx + y0 * vy);
        C = x0^2 + y0^2 - R^2;

        % Solve the quadratic equation
        discriminant = B^2 - 4 * A * C;

        if discriminant >= 0
            t1 = (-B + sqrt(discriminant)) / (2 * A);
            t2 = (-B - sqrt(discriminant)) / (2 * A);

            t_values = [t1, t2];
            t_values = t_values(t_values >= 0); % Keep only non-negative t values

            if ~isempty(t_values)
                points = zeros(length(t_values), 3);
                for i = 1:length(t_values)
                    points(i, :) = p_rotated + t_values(i) * v_rotated;
                end
                valid_points = points(points(:, 3) >= z_bottom & points(:, 3) <= z_top, :);
                valid_points = (inverse_rotation_matrix * valid_points')';
                
                num_valid_points = size(valid_points, 1);
                if num_valid_points > 0
                    impact_vector(current_index+1:current_index+num_valid_points, :) = valid_points;
                    impact_surface(current_index+1:current_index+num_valid_points, :) = 1; % 1 for cylindrical
                    current_index = current_index + num_valid_points;
                end
            end
        end

        % 2. Check intersection with the flat circular caps

        % Bottom cap (z = z_bottom)
        if vz ~= 0
            t_bottom = (z_bottom - z0_current) / vz;
            point_bottom = p_rotated + t_bottom * v_rotated;
            if t_bottom >= 0 && norm(point_bottom(1:2)) <= R
                point_bottom = (inverse_rotation_matrix * point_bottom')';
                current_index = current_index + 1;
                impact_vector(current_index, :) = point_bottom;
                impact_surface(current_index, :) = 3; % 3 for bottom cap
            end

            % Top cap (z = z_top)
            t_top = (z_top - z0_current) / vz;
            point_top = p_rotated + t_top * v_rotated;
            if t_top >= 0 && norm(point_top(1:2)) <= R
                point_top = (inverse_rotation_matrix * point_top')';
                current_index = current_index + 1;
                impact_vector(current_index, :) = point_top;
                impact_surface(current_index, :) = 2; % 2 for top cap
            end
        end
    end

    % Trim the preallocated arrays to remove unused space
    impact_vector = impact_vector(1:current_index, :);
    impact_surface = impact_surface(1:current_index, :);
end

function R = get_rotation_matrix(u, v)
    % Calculate the rotation matrix that rotates vector u to vector v
    % u: initial vector (3x1)
    % v: target vector (3x1)

    u = u / norm(u);
    v = v / norm(v);
    cross_uv = cross(u, v);
    dot_uv = dot(u, v);
    
    if norm(cross_uv) < 1e-6
        if dot_uv > 0
            R = eye(3); % No rotation needed
        else
            R = -eye(3); % 180 degree rotation
        end
    else
        skew_cross_uv = [0, -cross_uv(3), cross_uv(2); 
                         cross_uv(3), 0, -cross_uv(1); 
                         -cross_uv(2), cross_uv(1), 0];
        R = eye(3) + skew_cross_uv + skew_cross_uv^2 * ((1 - dot_uv) / norm(cross_uv)^2);
    end
end

function impact_vector_rphiz = convert_impact_data_to_cyl_coords(impact_vector_xyz, impact_surface, geom)
    % Estimate the number of impact points
    num_points = size(impact_vector_xyz, 1);
    
    % Preallocate output array for cylindrical coordinates [r, phi, z]
    impact_vector_rphiz = zeros(num_points, 3);
    
    % Normalize the axis vector
    axis_cyl = geom.axis_cyl;
    axis_cyl = axis_cyl / norm(axis_cyl);
    
    % Define the rotation matrix to align the cylinder axis with the z-axis
    z_axis = [0, 0, 1];
    rotation_matrix = get_rotation_matrix(axis_cyl, z_axis);

    % Initialize the index for filling the impact_vector_rphiz array
    current_index = 0;

    % Loop through each impact vector
    for ii = 1:num_points
        if ~isempty(impact_vector_xyz(ii, :))
            % Rotate the impact vector to align with the z-axis
            impact_rotated = (rotation_matrix * impact_vector_xyz(ii, :)')';

            % Initialize r, phi, z for the current impact point
            r = 0;
            phi = 0;
            z = impact_rotated(3);
            
            if impact_surface(ii) == 1  % 1 for cylindrical surface
                r = geom.R;
                phi = atan2(impact_rotated(2), impact_rotated(1)) * 180 / pi;
            elseif impact_surface(ii) == 2 || impact_surface(ii) == 3  % 2 for top cap, 3 for bottom cap
                r = sqrt(impact_rotated(1)^2 + impact_rotated(2)^2);
                phi = atan2(impact_rotated(2), impact_rotated(1)) * 180 / pi;
            end
            
            % Increment the index and store the result in the preallocated array
            current_index = current_index + 1;
            impact_vector_rphiz(current_index, :) = [r, phi, z];
        end
    end
    
    % Trim the array to remove any unused preallocated space
    impact_vector_rphiz = impact_vector_rphiz(1:current_index, :);
end

function [theta_bins, z_bins, heat_map] = accumulate_impact_vector(impact_rphiz, geom, weights, num_theta_bins, num_z_bins)
    % impact_rphiz: Nx3 matrix containing [r, phi, z] coordinates of the impact points
    % geom: Structure containing cylinder geometry with fields:
    %       - R: Radius of the cylinder
    %       - L: Length of the cylinder
    %       - z0: z-coordinate of the center of the cylinder
    % weights: Nx1 vector of weights corresponding to each impact point
    % num_theta_bins: Number of bins along the circumferential direction (theta)
    % num_z_bins: Number of bins along the height (z)
    
    % Calculate theta and z ranges
    theta_bins = linspace(-pi, pi, num_theta_bins+1); % Discretize theta from -π to π
    z_bottom = geom.z0 - geom.L / 2;
    z_top = geom.z0 + geom.L / 2;
    z_bins = linspace(z_bottom, z_top, num_z_bins+1); % Discretize z from z_bottom to z_top
    
    % Initialize the heat_map as a 2D matrix
    heat_map = zeros(num_z_bins, num_theta_bins);
    
    % Accumulate the weights into the heat_map
    for i = 1:size(impact_rphiz, 1)
        theta = deg2rad(impact_rphiz(i, 2)); % Convert phi to radians
        z = impact_rphiz(i, 3);
        
        % Find the corresponding bin indices
        [~, theta_idx] = histc(theta, theta_bins);
        [~, z_idx] = histc(z, z_bins);
        
        % Accumulate the weight into the appropriate bin
        if theta_idx > 0 && theta_idx <= num_theta_bins && z_idx > 0 && z_idx <= num_z_bins
            heat_map(z_idx, theta_idx) = heat_map(z_idx, theta_idx) + weights(i);
        end
    end
end

function plot_heat_map_2d(theta_bins, z_bins, heat_map)
    figure('color','w');
    imagesc(rad2deg(theta_bins(1:end-1)), z_bins, heat_map); % Convert theta_bins back to degrees for plotting
    set(gca, 'YDir', 'normal');
    xlabel('Theta (degrees)');
    ylabel('Z (height)');
    colormap(flipud(hot)); % Apply the flipped colormap to the current figure
    colorbar;
    xlim([-180,180])
    set(gca,'XTick',[-180:30:180])
end

function plot_heat_map_3d(fig_handle, theta_bins, z_bins, heat_map, geom)
    % Check if the figure handle is provided and is valid
    if nargin < 1 || ~isvalid(fig_handle)
        error('A valid figure handle must be provided.');
    end
    
    % Set the current figure to the provided handle
    figure(fig_handle);
    hold on;
    
    % Prepare the cylindrical coordinates
    [Theta, Z] = meshgrid(theta_bins(1:end-1), z_bins);
    X = geom.R * cos(Theta);
    Y = geom.R * sin(Theta);
    
    % Plot the heat map on the 3D cylinder surface
    surf(X, Y, Z, heat_map, 'EdgeColor', 'none');
    
    % Set labels and title if needed
    xlabel('U');
    ylabel('V');
    zlabel('W');
    title('Heat Map on 3D Cylinder Surface');
    colormap(flipud(hot)); % Apply the flipped colormap to the current figure
    colorbar;              % Display the colorbar    
    
    % Hold off the plot
    hold off;
end