function plot_LCFS_surface(haxis,R_lcfs,Z_lcfs,numAngles,face_alpha,face_color)
% nunAngles: Number of toroidal angles

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
hold on
surf(haxis,U3D, V3D, W3D, 'EdgeColor', 'none','FaceColor',face_color,'FaceAlpha',face_alpha)
hold off

% Formatting
font_size = 14;
set(gca,'FontSize',font_size)
xlabel('X [cm]','FontSize',font_size);
ylabel('Y [cm]','FontSize',font_size);
zlabel('Z [cm]','FontSize',font_size);
% xlim([-70,70]*2)
axis image
view([0,0])

end