% read in tracked quantities during debuging session:
close all
clear all
clc

% Import data:
dene = readtable('dene_track');
te = readtable("te_track");
ti = readtable("ti_track");
xi = readtable("xi_track");
yi = readtable("yi_track");
zi = readtable("zi_track");

dene = table2array(dene(:,2));
te = table2array(te(:,2));
ti = table2array(ti(:,2));
xi = table2array(xi(:,2));
yi = table2array(yi(:,2));
zi = table2array(zi(:,2));

figure
plot3(xi,dene,zi)
set(gca,'PlotBoxAspectRatio',[1,2,1])

figure
plot(xi,dene)
xlim([0,15])
ylim([-0.2,9]*1e12)
title('dene')

figure
subplot(1,2,1)
plot(xi,te)
xlim([0,15])
ylim([0,5])
title('te')

subplot(1,2,2)
plot(xi,ti)
xlim([0,15])
ylim([0,25])
title('ti')