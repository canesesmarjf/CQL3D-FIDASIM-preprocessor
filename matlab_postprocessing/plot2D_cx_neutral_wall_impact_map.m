function hfig = plot2D_cx_neutral_wall_impact_map(theta_bins, z_bins, heat_map)
    hfig = figure('color','w');
    imagesc(rad2deg(theta_bins(1:end-1)), z_bins, heat_map); % Convert theta_bins back to degrees for plotting
    set(gca, 'YDir', 'normal');
    xlabel('Theta (degrees)');
    ylabel('Z (height)');
    colormap(flipud(hot)); % Apply the flipped colormap to the current figure
    colorbar;
    xlim([-180,180])
    set(gca,'XTick',[-180:30:180])
end