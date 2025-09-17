% Step 1: 
% Get data
% Compare ne, Ti profiles in space and time

clear all
close all
clc

save_figure = 0;
output_dir = "Step_2_output/";

addpath(genpath("./matlab_postprocessing/"));

input_dir{1} = "./WHAM_test/test_1/" ; % FIDASIM beam, FIDASIM sink, nstop=50,
input_dir{2} = "./WHAM_test/test_2/" ; % FIDASIM beam, FREYA sink , nstop=50,
input_dir{3} = "./WHAM_test/test_3/" ; % FIDASIM beam, FIDASIM sink, nstop=50,
input_dir{4} = "/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_17_BEAM_EQDSK/2025_09_10/"; % NFREYA beam, NFREYA sink, nstop=600
% input_dir{5} = "./WHAM_test/test_4/" ; % FIDASIM beam, FIDASIM sink, nstop=50,
input_dir{5} = "./WHAM_test/" ; % FIDASIM beam, FIDASIM sink, nstop=50,

legend_text{1} = "FIDA beam, FIDA sink (DCX,HALOx5)";
legend_text{2} = "FIDA beam, FREYA sink";
legend_text{3} = "FIDA beam, FIDA sink (DCX)";
legend_text{4} = "FREYA beam, FREYA sink";
legend_text{5} = "FIDA beam, FIDA sink (DCX,HALOx4)";

opts.eion_max = 100; % [keV]
opts.ne_max = 1e14; % [cm^-3]
opts.te_max= 10; % [keV]
opts.pabs_max = 1e4; % [kW]
opts.tmax = 1200; % [ms]

% Get nc data:
filename_nc{1} = 'iaea2.2.nc';
filename_nc{2} = 'iaea2.2.nc';
filename_nc{3} = 'iaea2.2.nc';
filename_nc{4} = 'iaea.2.2.nc';
filename_nc{5} = 'iaea2.2.nc';

for ii = 1:numel(input_dir)
    pathname = input_dir{ii} + filename_nc{ii};
   
    % time:
    time = ncread(pathname,'/time');
    nc{ii}.nstop = find(time < 1e10,1,'last');
    tt_rng = 1:nc{ii}.nstop-1;
    nc{ii}.time = time(tt_rng);
    
    % Spatial grids:
    grid_r = ncread(pathname,'/solrz'); % [nz,nr]
    grid_z = ncread(pathname,'/solzz'); % [nz,nr]

    % Profiles:
    ne = ncread(pathname,'/densz1');
    nc{ii}.ne = permute(ne(1,:,:,tt_rng),[2 3 4 1]); % [nz,nr,nt]
    ke = ncread(pathname,'/energyz');
    nc{ii}.ke = permute(ke(:,2,:,tt_rng),[1 3 4 2]); % [nz,nr,nt]
    nc{ii}.ki = permute(ke(:,1,:,tt_rng),[1 3 4 2]); % [nz,nr,nt]

    % Absorbed power: NBI power is type=6
    pabs = ncread(pathname,'/powers_int'); %[type,species,nt] [W]
    nc{ii}.pabs_i = permute(pabs(6,1,tt_rng),[3,2,1]);
    nc{ii}.pabs_e = permute(pabs(6,2,tt_rng),[3,2,1]);
    nc{ii}.pabs = nc{ii}.pabs_i + nc{ii}.pabs_e;
end

%% PLOT: temporal profiles r=0, z=0

% DCX model:
% ======================
case_rng = [1,2,3,4,5]; 
hfig(1) = plot_basic_variables(nc,legend_text,case_rng,opts);

if save_figure
    save_fig(output_dir,"Beam deposition model")
end

%% Functions:

function hfig = plot_basic_variables(nc,legend_text,case_rng,opts)

color_vec = {"k","r","bl","g","m","c","r","g","bl","m","r","k","g","c","bl","r","k","m","g","c","bl","c"};

hfig = figure('color','w');
fontsize.latex = 15;
fontsize.legend = 15;
fontsize.axes = 12;
set(gcf,'Position',[258         607        1243         331])

% Ion energy:
subplot(1,4,1)
hold on
box on
for ii = case_rng
    y = permute(nc{ii}.ki(1,1,:),[3 2 1]);
    hki(ii) = plot(nc{ii}.time*1e3,y,color_vec{ii},'LineWidth',3);
end
set(gca,'FontSize',fontsize.axes)
hleg = legend(hki(case_rng),legend_text(case_rng));
set(hleg,"Location",'best','Interpreter','latex','FontSize',fontsize.legend)
xlabel("z [cm]",'Interpreter','latex','FontSize',fontsize.latex)
ylabel("Energy [keV]",'Interpreter','latex','FontSize',fontsize.latex)
title("Ion energy [keV]",'Interpreter','latex','FontSize',fontsize.latex)
ylim([0,opts.eion_max])
xlim([0,opts.tmax])

% Electron energy:
subplot(1,4,2)
hold on
box on
for ii = case_rng
    y = permute(nc{ii}.ke(1,1,:),[3 2 1]);
    hki(ii) = plot(nc{ii}.time*1e3,y,color_vec{ii},'LineWidth',3);
end
set(gca,'FontSize',fontsize.axes)
xlabel("time [ms]",'Interpreter','latex','FontSize',fontsize.latex)
ylabel("Energy [keV]",'Interpreter','latex','FontSize',fontsize.latex)
title("Electron energy [keV]",'Interpreter','latex','FontSize',fontsize.latex)
ylim([0,opts.te_max])
xlim([0,opts.tmax])

% Electron density:
subplot(1,4,3)
hold on
box on
for ii = case_rng
    y = permute(nc{ii}.ne(1,1,:),[3 2 1]);
    hki(ii) = plot(nc{ii}.time*1e3,y,color_vec{ii},'LineWidth',3);
end
set(gca,'FontSize',fontsize.axes)
xlabel("time [ms]",'Interpreter','latex','FontSize',fontsize.latex)
ylabel("[cm$^{-3}$]",'Interpreter','latex','FontSize',fontsize.latex)
title("Electron density [cm$^{-3}$]",'Interpreter','latex','FontSize',fontsize.latex)
ylim([0,opts.ne_max])
xlim([0,opts.tmax])

% if opt.include_turning_point
%     hold on
%     for ii = case_rng
%         [~,zzmax] = max(nc{ii}.ne{1}(1:40,1,end));
% 
% end

% NBI absorbed power:
subplot(1,4,4)
hold on
box on
for ii = case_rng
    y = nc{ii}.pabs*1e-3;
    hki(ii) = plot(nc{ii}.time*1e3,y,color_vec{ii},'LineWidth',3);
end
set(gca,'FontSize',fontsize.axes)
xlabel("time [ms]",'Interpreter','latex','FontSize',fontsize.latex)
ylabel("[kW]",'Interpreter','latex','FontSize',fontsize.latex)
title("NBI absorbed power [kW]",'Interpreter','latex','FontSize',fontsize.latex)
ylim([0,opts.pabs_max])
xlim([0,opts.tmax])
end

function save_fig(output_dir, file_name)

    figureName = [string(output_dir) + string(file_name)];

    % PDF figure:
    exportgraphics(gcf, figureName + ".pdf",'Resolution',600,'ContentType', 'vector') 

    % TIFF figure:
    exportgraphics(gcf, figureName + ".png",'Resolution',600) 

end