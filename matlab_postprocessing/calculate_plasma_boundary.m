function [phi,bflux,bflux_LCFS,R_lcfs,Z_lcfs] = ...
    calculate_plasma_boundary(plasma,fields,ne_edge)

    % This function calculates the plasma boundary as the LCFS given by a
    % edge plasma density ne_edge

    % Compute the magnetic flux:
    bflux = compute_magnetic_flux(fields); % use mesh(plasma.r,plasma.z,bflux') to plot
    
    % Obtain edge of plasma:
    R = plasma.r2d(:,1);
    zz_mid = round(size(plasma.dene,2)/2); % z index of midplane
    ne_R = plasma.dene(:,zz_mid); % Radial plasma profile at midplane (z=0m)
    rr_edge = find(ne_R > ne_edge,1,'last'); % r index for plasma edge
    r_edge = R(rr_edge); % Radius at plasma edge
    
    % Magnetic flux at the edge:
    bflux_LCFS = bflux(rr_edge,zz_mid); 
    
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
    Z_lcfs = plasma.z(iz);
    R_lcfs = plasma.r(ir);
end

function phi = compute_magnetic_flux(fields)
    
    % Extract the magnetic field components:
    Br = fields.br;
    Bz = fields.bz;

    % Grids: [m]
    r2d = fields.r2d*1e-2; % [m]
    z2d = fields.z2d*1e-2; % [m]
    
    % Get the size of the grid
    [nr, nz] = size(r2d);
    
    % Initialize the magnetic flux array
    phi = zeros(nr, nz);
    
    % Integrate Bz over the r direction
    for i = 2:nr
        dr = r2d(i, 1) - r2d(i-1, 1);
        r_prime = (r2d(i, 1) + r2d(i-1, 1)) / 2; % Midpoint of r interval
        phi(i, :) = phi(i-1, :) + Bz(i-1, :) * 2 * pi * r_prime * dr;
    end
end