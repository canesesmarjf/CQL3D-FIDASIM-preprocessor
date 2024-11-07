function plot_nbi_path(haxis,nbi_geom,nbi_path_length)

% Machine coordiantes:
% W = Z_cyl
% ^
% |
% |
% X----->U

uvw_src  = nbi_geom.src; % NBI source in machine coordinates:
uvw_axis = nbi_geom.axis; % NBI direction vector in machine coordinats:
adist    = nbi_geom.adist; % Aperture distance along axis:

% Calculate NBI path:
nbi_path = linspace(0,nbi_path_length);
uvw_nbi_path = uvw_src + uvw_axis*nbi_path;

% Aperture location:
uvw_aper = uvw_src + uvw_axis*adist;

% Plot NBI path:
hold on;
plot3(haxis,uvw_src(1),uvw_src(2),uvw_src(3),'go','MarkerSize',8)
plot3(haxis,uvw_aper(1),uvw_aper(2),uvw_aper(3),'ro','MarkerSize',8)
plot3(haxis,uvw_nbi_path(1,:),uvw_nbi_path(2,:),uvw_nbi_path(3,:),'k-','LineWidth',3)
hold off

end