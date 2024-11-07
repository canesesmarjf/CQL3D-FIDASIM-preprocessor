function plot3D_cx_neutral_wall_impact_map(fig_handle, theta_bins, z_bins, heat_map, vessel_geom)
    % Check if the figure handle is provided and is valid
    if nargin < 1 || ~isvalid(fig_handle)
        error('A valid figure handle must be provided.');
    end
    
    % Set the current figure to the provided handle
    figure(fig_handle);
    hold on;
    
    % Prepare the cylindrical coordinates
    [Theta, Z] = meshgrid(theta_bins(1:end-1), z_bins);
    X = vessel_geom.R * cos(Theta);
    Y = vessel_geom.R * sin(Theta);
    
    % Plot the heat map on the 3D cylinder surface
    surf(X, Y, Z, heat_map, 'EdgeColor', 'none');
    
    % Set labels and title if needed
    xlabel('U');
    ylabel('V');
    zlabel('W');
    title('CX neutral particle flux on wall [ions-cm^{-2}-s^{-1}]');
    colormap(flipud(hot)); % Apply the flipped colormap to the current figure
    colorbar;              % Display the colorbar    
    
    % Hold off the plot
    hold off;
end