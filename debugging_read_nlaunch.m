% READ nlaunch_<>.h5 file:

clear all
close all
clc

save_figure = 0;
suffix = "dcx";
% suffix = "halo";

addpath(genpath(...
    "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/matlab_postprocessing/"));
addpath(genpath(...
    "/home/jfcm/Repos/compx_matlab_toolkit"));

%% READ nlaunch HDF5 file:
file_name = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1b_standalone_runs/WHAM_Bob_IAEA_wall/";
file_name = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1b_standalone_runs/WHAM_test/";
file_name = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1a_coupled_runs/WHAM_test/";
file_name = file_name + "nlaunch" + "_" + suffix + ".h5";
info = h5info(file_name);

nlaunch = h5read(file_name,'/nlaunch');
papprox = h5read(file_name,'/papprox');
in_plasma = h5read(file_name,'/beam_grid/in_plasma');

% READ structured grid data:
grid.xmax = h5read(file_name,'/beam_grid/xmax');
grid.xmin = h5read(file_name,'/beam_grid/xmin');
grid.ymax = h5read(file_name,'/beam_grid/ymax');
grid.ymin = h5read(file_name,'/beam_grid/ymin');
grid.zmax = h5read(file_name,'/beam_grid/zmax');
grid.zmin = h5read(file_name,'/beam_grid/zmin');
grid.nx = h5read(file_name,'/beam_grid/nx');
grid.ny = h5read(file_name,'/beam_grid/ny');
grid.nz = h5read(file_name,'/beam_grid/nz');

% CREATE grid
grid = create_structured_grid(grid);

%% READ plasma equilibrium:
fidasim_run_dir = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1b_standalone_runs/WHAM_Bob_IAEA_wall/";
run_id = 'WHAM_Bob_IAEA_wall';
fidasim_run_dir = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1b_standalone_runs/WHAM_test/";
run_id = 'WHAM_test';
fidasim_run_dir = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1a_coupled_runs/WHAM_test/";
run_id = 'WHAM_test';
[plasma,fields] = read_fidasim_equilibrium(fidasim_run_dir,run_id);

%% COMPUTE plasma edge:

% Edge 1:
ne_edge = 3e12; % [cm^{-3}]
[phi,bflux,bflux_LCFS,R_lcfs,Z_lcfs] = ...
    calculate_plasma_boundary(...
    plasma,fields,ne_edge);

% Edge 2:
ne_edge = 1e8; % [cm^{-3}]
[phi,bflux,bflux_LCFS,R_lcfs2,Z_lcfs2] = ...
    calculate_plasma_boundary(...
    plasma,fields,ne_edge);

%% DEFINE circular vessel boundary:
R = 50;
theta = linspace(0,2*pi,1e2);
xcirc = R*cos(theta);
ycirc = R*sin(theta);

%% DEFINE circular plasma boundary:
Rplasma = 40;
xplasma = Rplasma*cos(theta);
yplasma = Rplasma*sin(theta);

%% PLOT IN_PLASMA at z = 0;

z_level = 0;
iz = find(grid.zc >= z_level,1);
figure('color','w'); 
hold on
val = squeeze(in_plasma(:,:,iz));
imagesc(grid.xc,grid.yc,val');
[xx,yy] = meshgrid(grid.xe,grid.ye);
plot(xx,yy,'k')
plot(xx',yy','k')
plot(xcirc,ycirc,'r','LineWidth',3)
plot(xplasma,yplasma,'c','LineWidth',3)
axis image
title("IN-PLASMA, z level  = " + z_level + " [cm]")
xlabel("x [cm]")
ylabel("y [cm]")
xlim([grid.xe(1),grid.xe(end)]*1.1)
ylim([grid.ye(1),grid.ye(end)]*1.1)

%% PLOT NLAUNCH at z = 0;

z_level = 0;
iz = find(grid.zc >= z_level,1);
figure('color','w'); 
hold on
val = squeeze(mean(nlaunch(:,:,iz-2:iz+2),3));
imagesc(grid.xc,grid.yc,val');
[xx,yy] = meshgrid(grid.xe,grid.ye);
plot(xx,yy,'k')
plot(xx',yy','k')
plot(xcirc,ycirc,'r','LineWidth',3)
plot(xplasma,yplasma,'c','LineWidth',3)
axis image
title(suffix + " NLAUNCH, z level  = " + z_level + " [cm]")
xlabel("x [cm]")
ylabel("y [cm]")
xlim([grid.xe(1),grid.xe(end)]*1.1)
ylim([grid.ye(1),grid.ye(end)]*1.1)

%% PLOT IN_PLASMA at x = 0:

x_level = 0;
ix = find(grid.xc >= x_level,1);
figure('color','w'); 
hold on
val = squeeze(in_plasma(ix,:,:));
imagesc(grid.yc,grid.zc,val');
[yy,zz] = meshgrid(grid.ye,grid.ze);
plot(yy,zz,'k')
plot(yy',zz','k')
axis image
title("IN-PLASMA, x level  = " + x_level + " [cm]")
xlabel("y [cm]")
ylabel("z [cm]")
xlim([grid.ye(1),grid.ye(end)]*1.1)
ylim([grid.ze(1),grid.ze(end)]*1.1)

hold on
plot(+R_lcfs,Z_lcfs,'r','LineWidth',2)
plot(-R_lcfs,Z_lcfs,'r','LineWidth',2)

plot(+R_lcfs2,Z_lcfs2,'m','LineWidth',2)
plot(-R_lcfs2,Z_lcfs2,'m','LineWidth',2)

set(gcf,'Position',[220,150,750,760])

%% PLOT NLAUNCH at x = 0:

x_level = 0;
ix = find(grid.xc >= x_level,1);
figure('color','w'); 
hold on
val = squeeze(mean(nlaunch(ix-2:ix+2,:,:),1));
imagesc(grid.yc,grid.zc,val');
[yy,zz] = meshgrid(grid.ye,grid.ze);
plot(yy,zz,'k')
plot(yy',zz','k')
axis image
title(suffix + " NLAUNCH, x level  = " + x_level + " [cm]")
xlabel("y [cm]")
ylabel("z [cm]")
xlim([grid.ye(1),grid.ye(end)]*1.1)
ylim([grid.ze(1),grid.ze(end)]*1.1)

hold on
plot(+R_lcfs,Z_lcfs,'r','LineWidth',2)
plot(-R_lcfs,Z_lcfs,'r','LineWidth',2)

plot(+R_lcfs2,Z_lcfs2,'m','LineWidth',2)
plot(-R_lcfs2,Z_lcfs2,'m','LineWidth',2)

set(gcf,'Position',[220,150,750,760])

%% PLOT NLAUNCH at y = 0:

y_level = 0;
iy = find(grid.yc >= y_level,1);
figure('color','w'); 
hold on
val = squeeze(mean(nlaunch(:,iy-2:iy+2,:),2));
imagesc(grid.xc,grid.zc,val');
[xx,zz] = meshgrid(grid.xe,grid.ze);
plot(xx,zz,'k')
plot(xx',zz','k')
axis image
title(suffix + " NLAUNCH, y level  = " + y_level + " [cm]")
xlabel("x [cm]")
ylabel("z [cm]")
xlim([grid.xe(1),grid.xe(end)]*1.1)
ylim([grid.ze(1),grid.ze(end)]*1.1)

hold on
plot(+R_lcfs,Z_lcfs,'r','LineWidth',2)
plot(-R_lcfs,Z_lcfs,'r','LineWidth',2)

plot(+R_lcfs2,Z_lcfs2,'m','LineWidth',2)
plot(-R_lcfs2,Z_lcfs2,'m','LineWidth',2)

set(gcf,'Position',[220,150,750,760])