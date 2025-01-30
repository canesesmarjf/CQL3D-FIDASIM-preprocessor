function plot_beam_grid(haxis,beam_grid,face_alpha)

hold on;
patch(haxis,'vertices', beam_grid.vertices_uvw', 'faces', beam_grid.faces', 'FaceColor', 'blue',...
    'FaceAlpha', face_alpha, 'EdgeColor', 'blue');
hold off

% Set axis properties
axis equal;
grid on;
view(3);

end