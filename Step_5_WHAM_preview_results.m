% Plot inputs for WHAM FIDASIM thermal calculation:

clear all
close all
clc

addpath(genpath("./matlab_postprocessing/"));

save_neutral_figs = 0;
save_cx_impact_figs = 0;
calc_impact_vectors = 0;
plot_plasma = 1;
plot_fields = 0;

scenario = 5.2;
scenario = 8;

switch scenario
    case 1
        run_id = "WHAM_example_1";
        fidasim_run_dir  = "./fidasim_files/" + run_id + "/";
        cql3d_run_dir = "./cql3d_files/" + run_id + "/";
    case 2

    case 3

    case 4
        run_id = "WHAM_no_f4d";
        fidasim_run_dir  = "./fidasim_files/" + run_id + "/";
        cql3d_run_dir = "./cql3d_files/" + run_id + "/";
    case 5
        run_id = "WHAM_low_ne_nonthermal";
        fidasim_run_dir  = "./fidasim_files/" + run_id + "/";
        cql3d_run_dir = "./cql3d_files/" + run_id + "/";
    case 5.1
        run_id = "WHAM_high_ne_nonthermal";
        fidasim_run_dir  = "./run_dir/" + run_id + "/";
        cql3d_run_dir = "./run_dir/" + run_id + "/";        
    case 5.2
        run_id = "WHAM_high_ne_nonthermal_multistep_cx";
        fidasim_run_dir  = "./run_dir/" + run_id + "/";
        cql3d_run_dir = "./run_dir/" + run_id + "/"; 
    case 5.3
        run_id = "WHAM_high_ne_thermal_multistep_cx";
        fidasim_run_dir  = "./run_dir/" + run_id + "/";
        cql3d_run_dir = "./run_dir/" + run_id + "/";    
    case 6
        run_id = "WHAM_low_ne_thermal";
        fidasim_run_dir  = "./fidasim_files/" + run_id + "/";
        cql3d_run_dir = "./cql3d_files/" + run_id + "/";      
    case 7
        run_id = "WHAM_wall_flux_cold_plasma";
        fidasim_run_dir  = "./fidasim_files/" + run_id + "/";
        cql3d_run_dir = "./cql3d_files/" + run_id + "/";  
    case 7.1
        run_id = "WHAM_wall_flux_cold_plasma_multistep_cx";
        fidasim_run_dir  = "./run_dir/" + run_id + "/";
        cql3d_run_dir = "./run_dir/" + run_id + "/";
    case 8
        run_id = "WHAM_NBI_kunal";
        fidasim_run_dir  = "./Step_1b_standalone_runs/" + run_id + "/";
        cql3d_run_dir = "./Step_1b_standalone_runs/" + run_id + "/";          
end

%% Get data:

% FIDASIM:
% =========================================================================

% FIDASIM input data:
[plasma,fields] = read_fidasim_equilibrium(fidasim_run_dir, run_id);
nbi_geom = read_fidasim_geometry(fidasim_run_dir, run_id);
inputs = read_fidasim_inputs(fidasim_run_dir, run_id);
beam_grid = calculate_fidasim_beam_grid(inputs);
config = read_config_files(fidasim_run_dir,cql3d_run_dir,run_id);

% TODO:
% The interpolation grid data () is to be found in "plasma"
% inter_grid.r   = plasma.r
% inter_grid.z   = plasma.z
% inter_grid.phi = plasma.phi (if it exists)
%
% TODO:
% The beam grid data can be found in "inputs"
% xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz 
% Or you can find the grid in birth or sink:
% x_grid or x
% y_grid or y
% z_grid or z

% Birth profile:
birth_file = fidasim_run_dir + run_id + "_birth.h5";
if exist(birth_file)==2
    birth = read_fidasim_sources("birth",fidasim_run_dir, run_id);
else
    disp("Birth data not available ...")
end

birth_file_1 = fidasim_run_dir + run_id + "_birth_1.h5";
if exist(birth_file_1)==2
    birth_1 = read_fidasim_sources("birth_1",fidasim_run_dir, run_id);
else
    disp("Birth_1 data not available ...")
end

% Sink Profile:
sink_file = fidasim_run_dir + run_id + "_sink.h5";
if exist(sink_file)==2
    sink = read_fidasim_sources("sink",fidasim_run_dir, run_id);
else
    disp("Sink data not available ...")
end

% CQL3D:
% =========================================================================
% Read CQLINPUT: ( we could get the NBI power from the run_id.dat file
% instead.
cqlinput_file = cql3d_run_dir + config.cql3d.cqlinput;
cqlinput_nml = read_nml(cqlinput_file);

% CQL3D mnemonic netCDF file:
cql_mnemonic_file = cql3d_run_dir + config.cql3d.plasma_file_name;
if exist(cql_mnemonic_file,'file')==2
    cql3d_nc = read_cql3d_mnemonic(cql3d_run_dir,config);
end

%% Derived quantities:

% NBI geometry:
% =========================================================================
% Machine coordiantes:
% W = Z_cyl
% ^
% |
% |
% X----->U

uvw_src  = nbi_geom.src; % NBI source in machine coordinates:
uvw_axis = nbi_geom.axis; % NBI direction vector in machine coordinats:
adist    = nbi_geom.adist; % Aperture distance along axis:

% Vacuum vessel dimensions:
% =========================================================================
vessel_geom.R = 50;   % Radius of the cylinder in [cm]
vessel_geom.L = 200;   % Length of the cylinder in [cm]
vessel_geom.z0 = 0.0;  % z-coordinate of the center of the cylinder
vessel_geom.axis_cyl = [0,0,1];

% Birth profile in velocity space:
% =========================================================================
if exist(birth_file)==2
    [birth_f, birth_vpar, birth_vper] = ...
        accumulate_source_on_velocity_grid...
        (birth.vpar, birth.vper, birth.weight, 50, 50);
end

if exist(birth_file_1)==2
    npar = 100;
    nper = 50;
    [birth_1_f, birth_1_vpar, birth_1_vper] = ...
        accumulate_source_on_velocity_grid...
        (birth_1.vpar, birth_1.vper, birth_1.weight, npar, nper);
end

% Sink profile in velocity space:
% =========================================================================
if exist(sink_file)==2
    npar = 70;
    nper = 35;
    [sink_f, sink_vpar, sink_vper] = ...
        accumulate_source_on_velocity_grid...
        (sink.vpar, sink.vper, sink.weight, npar, nper);
end

% CX neutrals on the wall:
% =========================================================================
if exist(sink_file)==2
    if calc_impact_vectors    
        [theta_bins, z_bins, particle_flux_map, energy_flux_map] = ...
        calculate_cx_neutral_wall_impact_map(vessel_geom, sink, 100, 81);
    end
end

% Plasma edge:
% =========================================================================
[phi,bflux,bflux_LCFS,R_lcfs,Z_lcfs] = ...
    calculate_plasma_boundary(plasma,fields,1e12);

%% Print NBI absorption diagnostic:

if exist(birth_file)==2

    % Total power in ions deposited in [W]:
    Pabs = sum(birth.energy.*birth.weight*1e3*e_c); % [W]
    Pinj = inputs.pinj*1e6; % [W]
    shinethrough = 1 - Pabs/Pinj;
    disp(" ")
    disp("(FIDASIM) Total NBI power injected: " + num2str(Pinj*1e-3,4) + " [kW]")
    disp("(FIDASIM) Total NBI power absorbed: " + num2str(Pabs*1e-3,4) + " [kW]")
    disp("(FIDASIM) " + num2str(shinethrough*100,4) + "% shine-through")
    
    % CQL3D NBI power:
    if exist(cql_mnemonic_file)==2
        
        % Get NBI power from netCDF file:
        powers_int = cql3d_nc.powers_int; % 'powers(*,6,k,t)=Ion particle source'
        nbi_power_cql3d = powers_int(end,1,end); % [W]
        Pinj_cql = cqlinput_nml.frsetup.bptor{1};

        shinethrough = 1 - nbi_power_cql3d/Pinj_cql;
        disp(" ")
        disp("(CQL3D-M) Total NBI power injected: " + num2str(Pinj_cql*1e-3,4) + " [kW]")
        disp("(CQL3D-M) Total NBI power absorbed: " + num2str(nbi_power_cql3d*1e-3,4) + " [kW]")
        disp("(CQL3D-M) " + num2str(shinethrough*100,4) + "% shine-through")

    end
end

%% Plot plasma:

if plot_plasma
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
end
%% Plot fields:

if plot_fields
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
end

%% Plot ion sink profile in velocity space:

if exist(sink_file)==2
        
    hfig = figure('color','w');
    box on
    hold on
    surf(sink_vpar, sink_vper, sink_f','LineStyle','none');
    surf(birth_vpar, birth_vper, 1*birth_f','LineStyle','none');

    % Plot beam energies:
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

    if save_cx_impact_figs
        save_fig(hfig,fidasim_run_dir,"cx_ion_sink_profile"); 
    end
end

%% Plot ion birth_1 profile in velocity space:

if exist(birth_file_1)==2
        
    hfig = figure('color','w');
    box on
    hold on
    surf(birth_1_vpar, birth_1_vper, birth_1_f','LineStyle','none');
    surf(birth_vpar, birth_vper, birth_f','LineStyle','none');

    % Plot beam energies:
    [e_levels,~,~] = unique(birth.energy);
    ie = 1:10:numel(birth.energy);
    hnbi = plot3(birth.vpar(ie), birth.vper(ie), max(birth_1_f(:))*ones(size(ie)),'go');
    hnbi.MarkerEdgeColor = 'g';
    hnbi.MarkerFaceColor = hnbi.MarkerEdgeColor;

    set(gca, 'YDir', 'normal');
    xlabel('$v_{\parallel}$ [cm/s]','Interpreter','latex','FontSize',18);
    ylabel('$v_{\perp}$ [cm/s]','Interpreter','latex','FontSize',18);
    title({'Ion birth 1st gen profile';'$f_n(\vec{v})\int{ f_+(\vec{v}'')\langle{\sigma v_{rel}}\rangle_\phi^{cx} d^3 v''}$'},'Interpreter','latex','FontSize',16); % (\vec{v},\vec{v}'')
    set(gca,'ZScale','lin')
    colormap(flipud(hot))
    caxis([0,max(birth_1_f(:))])
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
        hline(ii)=plot3(vel_vec(ii)*cos(theta),vel_vec(ii)*sin(theta),max(birth_1_f(:))*ones(size(theta)),line_color{ii},'LineWidth',2);
        legend_text{ii} = num2str(energy_vec(ii)*1e-3,3) + " [keV]";
    end
    legend_text{ii+1} = "NBI";
    legend([hline,hnbi],legend_text)

    vmax = vel_vec(3);
    xlim([-vmax,vmax]*1.2)
    ylim([    0,vmax]*1.2)

    % Total power in ions deposited in [W]:
    Pbirth1 = sum(birth_1.energy.*birth_1.weight*e_c)*1e3 % [W];
    disp(" ")
    disp("(FIDASIM) Total power absorbed by DCX: " + num2str(Pbirth1,4) + " [W]")

    % Ratio between sink current and birth current:
    r_cx01 = sum(sink.weight) / (sum(birth.weight) + sum(birth_1.weight));
    disp(" ")
    disp("(FIDASIM) r_cx01 = sum(sink.w)/(sum(birth.w)+sum(birth_1.w)): " + num2str(r_cx01,3))

    if save_cx_impact_figs
        save_fig(hfig,fidasim_run_dir,"DCX_ion_birth_1_profile"); 
    end
end

%% Plot cx neutral flux to wall for different energy ranges:
if exist(sink_file)==2
    if calc_impact_vectors  
        max_energy = max(sink.energy);
        min_energy = min(sink.energy);
        
        hfig = figure('color','w');
        [fE,fE_edges] = histcounts(sink.energy,'BinLimits',[min_energy,max_energy],'Normalization','pdf');
        plot(fE_edges(1:end-1),fE,'LineWidth',2)
        title('CX neutral energy PDF')
        xlabel('Energy [keV]')

        energy_rng.min   = [0  , 0.1, 0.2,0.3]*max_energy;
        energy_rng.max   = [0.1, 0.2, 0.3,1.0]*max_energy;
        energy_rng.color = {'r','bl','g' ,'m'};

        % Loop through each energy range and add a shaded area
        for i = 1:length(energy_rng.min)
            % Define the x and y coordinates for the shaded area
            x_patch = [energy_rng.min(i), energy_rng.max(i), energy_rng.max(i), energy_rng.min(i)];
            y_patch = [0, 0, max(fE), max(fE)];
            
            % Add a shaded patch with semi-transparency
            patch(x_patch, y_patch, energy_rng.color{i}, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end

        if save_cx_impact_figs
            save_fig(hfig,fidasim_run_dir,"cx_energy_PDF"); 
        end 

        for ee = 1:numel(energy_rng.max)
            rng = find(sink.energy >= energy_rng.min(ee) & sink.energy < energy_rng.max(ee));
            sink_subset{ee}.energy = sink.energy(rng);
            sink_subset{ee}.weight = sink.weight(rng);
            sink_subset{ee}.n_sink = numel(rng);
            sink_subset{ee}.U = sink.U(rng);
            sink_subset{ee}.V = sink.V(rng);
            sink_subset{ee}.W = sink.W(rng);
            sink_subset{ee}.vU = sink.vU(rng);
            sink_subset{ee}.vV = sink.vV(rng);
            sink_subset{ee}.vW = sink.vW(rng);

            % Calculate cx neutral fluxes on wall:
            [theta_bins, z_bins, particle_flux_map, energy_flux_map] = ...
            calculate_cx_neutral_wall_impact_map(vessel_geom, sink_subset{ee}, 100, 81);
        
            % Plot the neutral particle flux density map:
            [hfig] = plot2D_cx_neutral_wall_impact_map(theta_bins, z_bins, particle_flux_map);
            title('CX neutral flux density on vaccum vessel [atom/cm^2/s]');

            % Define the energy range label
            energy_label = sprintf('Energy: [%.3f, %.3f] keV', energy_rng.min(ee), energy_rng.max(ee));
            
            % Add the label at the upper left corner inside the plot area
            x_pos = -170; % Adjust the x position as needed
            y_pos = max(z_bins) - (max(z_bins) - min(z_bins)) * 0.05; % Slightly below the top edge
            text(x_pos, y_pos, energy_label, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black', 'HorizontalAlignment', 'left');

            if save_cx_impact_figs
                save_fig(hfig,fidasim_run_dir,"cx_energy_flux_map_2D_rng_" + num2str(ee)); 
            end  
        end

    end
end
%% Plot CX neutral impact maps:

if exist(sink_file)==2
    if calc_impact_vectors    
        % Calculate cx neutral fluxes on wall:
        [theta_bins, z_bins, particle_flux_map, energy_flux_map,P_wall] = ...
        calculate_cx_neutral_wall_impact_map(vessel_geom, sink, 100, 100);
    
        % Plot the neutral particle flux density map:
        hfig = plot2D_cx_neutral_wall_impact_map(theta_bins, z_bins, particle_flux_map);
        title('CX neutral flux density on vaccum vessel [atom/cm^2/s]');
    
        if save_cx_impact_figs
            save_fig(hfig,fidasim_run_dir,"cx_neutral_particle_flux_map_2D"); 
        end

        % Plot power density map:
        hfig = plot2D_cx_neutral_wall_impact_map(theta_bins, z_bins, energy_flux_map);
        title('CX energy flux density on vaccum vessel [W/cm^2]');

        if save_cx_impact_figs
            save_fig(hfig,fidasim_run_dir,"cx_energy_flux_map_2D"); 
        end        

        % Print power intercepted by the wall:
        disp("(FIDASIM) Neutral flux power lost to wall: " + num2str(P_wall.cyl*1e-3,4) + " [kW]")

    end
end

%% Plot geometry:

h3d = figure('color','w');

% Plot Plasma edge as a surface:
face_alpha = 0.2;
numAngles = 30;
plot_LCFS_surface(gca,R_lcfs,Z_lcfs,numAngles,face_alpha)

% Plot NBI optical axis:
nbi_path_length = 300;
plot_nbi_path(gca,nbi_geom,nbi_path_length)

% Plot beam grid:
face_alpha = 0.2;
plot_beam_grid(gca,beam_grid,face_alpha)

% Optionally, adjust the view and lighting for better visualization
camlight;
lighting gouraud;

% Add birth points:
if exist(birth_file)==2
    hold on
    plot3(birth.U,birth.V,birth.W,'k.')
    hold off
end

if exist(birth_file_1)==2
    hold on
    plot3(birth_1.U,birth_1.V,birth_1.W,'r.','MarkerSize',3)
    hold off
end

% Add sink points:
if exist(sink_file)==2
    hold on
    plot3(sink.U,sink.V,sink.W,'g.')
    hold off
end

% Add cx power density impacting vaccum vessel:
if calc_impact_vectors && (exist(sink_file)==2)
    flux_wall = particle_flux_map;

    plot3D_cx_neutral_wall_impact_map...
        (h3d, theta_bins, z_bins, flux_wall, vessel_geom)
    
    caxis([0,0.98*max(flux_wall(:))])
    lighting none
end

% Add grid: (Needs work, it is not working)
if 0
    [UU,WW] = meshgrid(birth.U_grid,birth.W_grid);
    plot3(WW,zeros(size(WW)),UU,'k')
    plot3(WW',zeros(size(WW')),UU','k')
end

%% Plot ne and nn radial profiles:

figure('color','w')
set(gcf,'Position',[303   445   937   517])

subplot(1,3,1)
hold on
box on
contourf(plasma.r2d,plasma.z2d,plasma.dene,20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('Electron density [cm^{-3}]')
xlabel('R [cm]')
zlabel('Z [cm]')

subplot(1,3,2)
hold on
box on
contourf(plasma.r2d,plasma.z2d,plasma.denn,20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('Neutral density [cm^{-3}]')
xlabel('R [cm]')
zlabel('Z [cm]')

subplot(1,3,3)

units = '[Pa]';
units = '[mTorr]';
T_neutral = 1; % [eV]
switch units
    case '[Pa]'
        neutral_pressure = plasma.denn*10*e_c*1e6;
    case '[mTorr]'
        neutral_pressure = plasma.denn*10*e_c*1e6*7.5;
end

hold on
box on
contourf(plasma.r2d,plasma.z2d,neutral_pressure,20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title("Neutral pressure " + units)
xlabel('R [cm]')
zlabel('Z [cm]')
text(8, 80, "T_{n} = " + num2str(T_neutral) + " [eV]", ...
    'Color', 'white', 'FontSize', 14, ...
    'BackgroundColor', 'none');
% 1 Pa = 7.5 mTorr
% Thus 0.2 Pa = 1.5 mTorr
hf = gcf;
set(hf.Children,'FontSize',12)

figure
hold on
box on
contourf(plasma.r2d,plasma.z2d,plasma.te,20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('Electron temperture [keV]')
xlabel('R [cm]')
zlabel('Z [cm]')

hf = gcf;
set(hf.Children,'FontSize',12)

%% Plot radial dene and denn profiles:

iz_midplane = round(numel(plasma.z)/2);
dene_r_profile = plasma.dene(:,iz_midplane);
denn_r_profile = plasma.denn(:,iz_midplane);
te_r_profile = plasma.te(:,iz_midplane);
ti_r_profile = plasma.ti(:,iz_midplane);

fctr(1) = 10e2;

% Compute everlap:
overlap_r_profile = dene_r_profile.*denn_r_profile;
overlap_r_profile = overlap_r_profile/max(overlap_r_profile);
fctr(2) = 0.1*max(dene_r_profile)/max(overlap_r_profile);
fctr(3) = 1e13;
fctr(4) = 1e12;

figure('color','w')
box on
hold on
hr(1) = plot(plasma.r,dene_r_profile,'k');
hr(2) = plot(plasma.r,denn_r_profile*fctr(1),'r');
hr(3) = plot(plasma.r,...
    overlap_r_profile*fctr(2),'m');
hr(4) = plot(plasma.r,te_r_profile*fctr(3),'g');
hr(5) = plot(plasma.r,ti_r_profile*fctr(4),'bl');

legend_text{1} = "dene";
legend_text{2} = "denn x " + fctr(1);
legend_text{3} = "dene x denn";
legend_text{4} = "te x " + fctr(3)*1e-13 + "e13";
legend_text{5} = "ti x " + fctr(3)*1e-13 + "e12";

set(hr,'lineWidth',3)
ylabel("density [cm^{-3}]")
xlabel("R [cm]")
legend(hr,legend_text)
set(gca,'FontSize',14)
clear hr

%% Convert birth profiles to cylindrical:
[X_grid, Y_grid, Z_grid] = meshgrid(birth.y, birth.z, birth.x);

% Transform to cylindrical coordinates
n_r = 50; n_phi = 100; n_z = 50;
R_new = linspace(0, max(sqrt(birth.y.^2 + birth.z.^2)), n_r);
Phi_new = linspace(-pi, pi, n_phi);
Z_new = linspace(min(birth.x), max(birth.x), n_z);
[R_cyl, Phi_cyl, Z_cyl] = meshgrid(R_new, Phi_new, Z_new);

% Interpolate data to cylindrical grid [phi,r,z]
data = permute(birth.dens_full,[2 3 1]);
data_cyl = interp3(X_grid, Y_grid, Z_grid, data, ...
                   R_cyl .* cos(Phi_cyl), R_cyl .* sin(Phi_cyl), Z_cyl, ...
                   'linear', 0);

figure('color','w');
subplot(1,2,1)
hold on 
box on
birth_RZ = permute(mean(data_cyl,1),[2,3,1]);
contourf(R_new,Z_new,(birth_RZ'),20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('0th gen ION BIRTH [cm^{-3}s^{-1}]')
xlabel('R [cm]')
zlabel('Z [cm]')
max_birth_rate = max(birth_RZ(:));
caxis([0,max_birth_rate])
set(gca,'FontSize',12)

subplot(1,2,2)

% Interpolate data to cylindrical grid
data = permute(birth_1.dens_dcx,[2 3 1]);
data_cyl = interp3(X_grid, Y_grid, Z_grid, data, ...
                   R_cyl .* cos(Phi_cyl), R_cyl .* sin(Phi_cyl), Z_cyl, ...
                   'linear', 0);

hold on 
box on
birth_1_RZ = permute(mean(data_cyl,1),[2,3,1]);
contourf(R_new,Z_new,(birth_1_RZ'),20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('1st gen DCX ION BIRTH [cm^{-3}s^{-1}]')
xlabel('R [cm]')
zlabel('Z [cm]')
set(gca,'FontSize',12)

% Interpolate data to cylindrical grid [phi,r,z]
data = permute(sink.dens,[4 3 2 1]);
data_cyl = interp3(X_grid, Y_grid, Z_grid, data, ...
                   R_cyl .* cos(Phi_cyl), R_cyl .* sin(Phi_cyl), Z_cyl, ...
                   'linear', 0);
hf = gcf;
set(hf.Children,'FontSize',12)

figure('color','w');
subplot(1,2,1)

hold on 
box on
sink_RZ = permute(mean(data_cyl,1),[2,3,1]);
contourf(R_new,Z_new,(sink_RZ'),20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('0th gen ION SINK [cm^{-3}s^{-1}]')
xlabel('R [cm]')
zlabel('Z [cm]')
caxis([0,max_birth_rate])
set(gca,'FontSize',12)

subplot(1,2,2)

hold on 
box on
contourf(R_new,Z_new,(birth_RZ-sink_RZ)',20,'LineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('Impact ionization rate [cm^{-3}s^{-1}]')
xlabel('R [cm]')
zlabel('Z [cm]')
% caxis([0,max_birth_rate/10])
set(gca,'FontSize',12)

% Ionization rate ceofficient:
iz_coeff = 1.7e-9; % Te=2keV, En = 1eV from FIDASIM atomic tables [cm^{3}s^{-1}]
iz_rate = iz_coeff*(1e10); % [s^{-1}]
iz_source = iz_coeff*max(dene_r_profile.*denn_r_profile); % [e-cm^{-3}s^{-1}]
disp("analytical IZ rate: " + num2str(iz_source*1e-14) + "x1e14 [e-cm^{-3}s^{-1}]")

hf = gcf;
set(hf.Children,'FontSize',12)
%% Test plots: Birth and sink U_V slices
% Plasma edge:
R_edge = R_lcfs(round(numel(R_lcfs)/2));
Y_edge = R_edge*cos(linspace(0,2*pi));
Z_edge = R_edge*sin(linspace(0,2*pi));

% Birth profile; YZ
figure('color','w'); 
box on
hold on
z_value = mean(permute(birth.dens_full(30:50,:,:),[2 3 1]),3);
contourf(birth.y,birth.z,(z_value))
plot(Y_edge,Z_edge,'g','LineWidth',3)
axis image
colormap(flipud(hot))
colorbar
xlim([-1,1]*20)
ylim([-1,1]*20)
title('0th gen edge neutral ION BIRTH points U-V plane')

% Sink profile; YZ
figure('color','w'); 
box on
hold on
z_value = mean(permute(sink.dens(1,30:50,:,:),[3 4 2 1]),3);
contourf(birth.y,birth.z,(z_value))
plot(Y_edge,Z_edge,'g','LineWidth',3)
axis image
colormap(flipud(hot))
colorbar
xlim([-1,1]*20)
ylim([-1,1]*20)
title('0th gen edge neutral ION SINK points U-V plane')

% Birth_1 profile; YZ
figure('color','w'); 
hold on
box on
z_value = mean(permute(birth_1.dens_dcx(30:50,:,:),[2 3 1]),3);
contourf(birth.y,birth.z,(z_value))
plot(Y_edge,Z_edge,'g','LineWidth',3)
axis image
colormap(flipud(hot))
colorbar
xlim([-1,1]*20)
ylim([-1,1]*20)
title('1st gen DCX ION BIRTH points U-V plane')

%% Neutral densities:

% ====================================================
% The following code needs to be made into functions:
% ====================================================

info = h5info(fidasim_run_dir + run_id + "_neutrals.h5");
disp(" ")
disp(" Neutral densities: ")

% Diagnostics:
if 0
    disp({info.Groups.Name})
    disp({info.Groups(2).Datasets.Name})
    disp(info.Groups(2).Datasets(1).Dataspace)
end

denn_full_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/full/dens');
denn_half_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/half/dens');
denn_third_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/third/dens');

try
    denn_dcx_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/dcx/dens');
end

grid_x = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/grid/x_grid');
grid_y = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/grid/y_grid');
grid_z = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/grid/z_grid');

% These have dimensions of 70,50,50 (X,Y,Z)
denn_full = permute(sum(denn_full_n,1),[2,3,4,1]);
denn_half = permute(sum(denn_half_n,1),[2,3,4,1]);
denn_third = permute(sum(denn_third_n,1),[2,3,4,1]);

try
    denn_dcx = permute(sum(denn_dcx_n,1),[2,3,4,1]);
end

denn_full_XZ = permute(sum(denn_full,2),[1,3,2]);
denn_half_XZ = permute(sum(denn_half,2),[1,3,2]);
denn_third_XZ = permute(sum(denn_third,2),[1,3,2]);

try
    denn_dcx_XZ = permute(sum(denn_dcx,2),[1,3,2]);
end

caxis_min = 6.0;
caxis_max = 10.5;

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_full_XZ"); 
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_half_XZ"); 
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_third_XZ"); 
end

try
    hfig = figure('color','w');
    hold on
    logZ = log10(denn_dcx_XZ);
    levels = 80;
    line_style = 'none';
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
        save_fig(hfig,fidasim_run_dir,"neutral_density_dcx_XZ"); 
    end
end

%%  Plot cylindrical neutral gas densities:

try
data = permute(denn_dcx,[2 3 1]);
data_cyl = interp3(X_grid, Y_grid, Z_grid, data, ...
                   R_cyl .* cos(Phi_cyl), R_cyl .* sin(Phi_cyl), Z_cyl, ...
                   'linear', 0);

figure('color','w');
hold on 
box on
birth_RZ = permute(mean(data_cyl,1),[2,3,1]);
contourf(R_new,Z_new,(birth_RZ'),20,'lineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('DCX neutral density [cm^{-3}]')
xlabel('R [cm]')
zlabel('Z [cm]')
max_birth_rate = max(birth_RZ(:));
caxis([0,max_birth_rate])
set(gca,'FontSize',12)
end

try
data = permute(denn_full,[2 3 1]);
data_cyl = interp3(X_grid, Y_grid, Z_grid, data, ...
                   R_cyl .* cos(Phi_cyl), R_cyl .* sin(Phi_cyl), Z_cyl, ...
                   'linear', 0);

figure('color','w');
hold on 
box on
birth_RZ = permute(mean(data_cyl,1),[2,3,1]);
contourf(R_new,Z_new,(birth_RZ'),20,'lineStyle','none')
plot(R_lcfs,Z_lcfs,'g','LineWidth',3)
% axis image
set(gca,'PlotBoxAspectRatio',[1,2,1])
colormap(flipud(hot))
colorbar
xlim([0,1]*20)
ylim([-1,1]*100)
title('Full neutral density [cm^{-3}]')
xlabel('R [cm]')
zlabel('Z [cm]')
max_birth_rate = max(birth_RZ(:));
caxis([0,max_birth_rate])
set(gca,'FontSize',12)
end

%% Halo and DCX neutral densities:

try
    denn_dcx_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/dcx/dens');
    denn_halo_n = h5read(fidasim_run_dir + run_id + "_neutrals.h5",'/halo/dens');
    denn_dcx  = permute(sum(denn_dcx_n ,1),[2,3,4,1]);
    denn_halo = permute(sum(denn_halo_n,1),[2,3,4,1]);
    denn_dcx_XZ  = permute(sum(denn_dcx ,2),[1,3,2]);
    denn_halo_XZ = permute(sum(denn_halo,2),[1,3,2]);
catch
    disp("DCX and halo densities not available ...")
    return
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_NBI_XZ"); 
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_dcx_XZ"); 
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_halo_XZ"); 
end

hfig = figure('color','w');
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
    save_fig(hfig,fidasim_run_dir,"neutral_density_dhalo_XZ"); 
end

%% Distribution:

%% Beam grid plotting:
% Would be good to plot the beam grid to inspect the cell size relative to
% the plasma size

%% Functions:

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



