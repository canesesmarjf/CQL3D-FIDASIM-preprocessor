function [theta_bins, z_bins, particle_flux_map, energy_flux_map] = ...
    calculate_cx_neutral_wall_impact_map(vessel_geom, sink, nz, ntheta)

    % This function takes in the sink source, the vacuum vessel geometry
    % and computes the neutral partcle flux density and energy flux density
    % intercepted by the vacuuum vessel wall from CX neutrals emitted from
    % the plasma

    % Compute the impact points in cartesian coordinates:
    % =========================================================================
    
    % Package the sink vectors:
    p_vec = zeros(sink.n_sink,3);
    v_vec = zeros(sink.n_sink,3);
    for ii = 1:sink.n_sink
        p_vec(ii,:) = [sink.U(ii),sink.V(ii),sink.W(ii)];
        v_vec(ii,:) = [sink.vU(ii),sink.vV(ii),sink.vW(ii)];
    end

    % Compute the position vectors of the neutrals hitting the wall:
    impact_vector = zeros(sink.n_sink,3);
    impact_surface = zeros(sink.n_sink,3);
    [impact_vector,impact_surface] = calculate_impact_vector_on_cylinder(p_vec,v_vec,vessel_geom);
    
    % Convert impact vectors to cylindrical coordinates:
    % =========================================================================
    impact_vector_rphiz = convert_impact_data_to_cyl_coords(impact_vector, impact_surface, vessel_geom);
    
    % Calculate CX impact map on vessel:
    % =========================================================================
    ii_rng = find(impact_surface == 1);
    [theta_bins, z_bins, cx_neutral_flux_wall] = accumulate_impact_vector(impact_vector_rphiz(ii_rng,:), vessel_geom, sink.weight(ii_rng), ntheta, nz);
    dz = mean(diff(z_bins)); % [cm]
    ds = vessel_geom.R*mean(diff(theta_bins)); % [cm]
    particle_flux_map = cx_neutral_flux_wall/(dz*ds); % [ion-cm^{-2}/s]

    % Calculate power deposition on vaccum vessel:
    % =========================================================================
    marker_weight =  sink.weight(ii_rng).* sink.energy(ii_rng)*(1e3)*e_c; % [W/s]
    [~, ~, cx_energy_flux_wall] = accumulate_impact_vector(impact_vector_rphiz(ii_rng,:), vessel_geom, marker_weight, ntheta, nz);
    energy_flux_map = cx_energy_flux_wall/(dz*ds); % [W-cm^{-2}]
end

function [impact_vector, impact_surface] = calculate_impact_vector_on_cylinder(p, v, vessel_geom)
    % p: Matrix of starting points [N x 3]
    % v: Matrix of velocity vectors [N x 3]
    % vessel_geom: Structure containing the cylinder geometry with fields:
    %       - R: Radius of the cylinder
    %       - L: Length of the cylinder along the z-axis
    %       - z0: z-coordinate of the center of the cylinder
    %       - axis_cyl: Unit vector representing the direction of the cylinder's axis
    % impact_vector: Matrix of intersection points [M x 3]
    % impact_surface: Array of integers indicating which surface was intersected:
    %                 1 for cylindrical, 2 for top cap, 3 for bottom cap

    % Normalize the axis vector
    axis_cyl = vessel_geom.axis_cyl;
    axis_cyl = axis_cyl / norm(axis_cyl);
    
    % Define the rotation matrix to align the cylinder axis with the z-axis
    z_axis = [0, 0, 1];
    rotation_matrix = get_rotation_matrix(axis_cyl, z_axis);
    inverse_rotation_matrix = rotation_matrix'; % Transpose of the rotation matrix

    % Extract the cylinder parameters from the vessel_geom structure
    R = vessel_geom.R;
    L = vessel_geom.L;
    z_center = vessel_geom.z0;

    % Calculate the z-coordinate of the bottom and top bases
    z_bottom = z_center - L/2;
    z_top = z_center + L/2;

    % Estimate the maximum number of potential intersections
    max_intersections = 2 * size(p, 1);  % Two intersections per vector as an estimate

    % Preallocate arrays
    impact_vector = zeros(max_intersections, 3);
    impact_surface = zeros(max_intersections, 1);

    current_index = 0;

    % Loop through each pair of p and v
    for ii = 1:size(p, 1)
        % Rotate the points and vectors to align with the z-axis
        p_rotated = (rotation_matrix * p(ii, :)')';
        v_rotated = (rotation_matrix * v(ii, :)')';

        % Unpack the rotated vector components
        x0 = p_rotated(1);
        y0 = p_rotated(2);
        z0_current = p_rotated(3); % This is the z-coordinate of the starting point
        
        vx = v_rotated(1);
        vy = v_rotated(2);
        vz = v_rotated(3);

        % 1. Check intersection with the curved cylindrical surface
        A = vx^2 + vy^2;
        B = 2 * (x0 * vx + y0 * vy);
        C = x0^2 + y0^2 - R^2;

        % Solve the quadratic equation
        discriminant = B^2 - 4 * A * C;

        if discriminant >= 0
            t1 = (-B + sqrt(discriminant)) / (2 * A);
            t2 = (-B - sqrt(discriminant)) / (2 * A);

            t_values = [t1, t2];
            t_values = t_values(t_values >= 0); % Keep only non-negative t values

            if ~isempty(t_values)
                points = zeros(length(t_values), 3);
                for i = 1:length(t_values)
                    points(i, :) = p_rotated + t_values(i) * v_rotated;
                end
                valid_points = points(points(:, 3) >= z_bottom & points(:, 3) <= z_top, :);
                valid_points = (inverse_rotation_matrix * valid_points')';
                
                num_valid_points = size(valid_points, 1);
                if num_valid_points > 0
                    impact_vector(current_index+1:current_index+num_valid_points, :) = valid_points;
                    impact_surface(current_index+1:current_index+num_valid_points, :) = 1; % 1 for cylindrical
                    current_index = current_index + num_valid_points;
                end
            end
        end

        % 2. Check intersection with the flat circular caps

        % Bottom cap (z = z_bottom)
        if vz ~= 0
            t_bottom = (z_bottom - z0_current) / vz;
            point_bottom = p_rotated + t_bottom * v_rotated;
            if t_bottom >= 0 && norm(point_bottom(1:2)) <= R
                point_bottom = (inverse_rotation_matrix * point_bottom')';
                current_index = current_index + 1;
                impact_vector(current_index, :) = point_bottom;
                impact_surface(current_index, :) = 3; % 3 for bottom cap
            end

            % Top cap (z = z_top)
            t_top = (z_top - z0_current) / vz;
            point_top = p_rotated + t_top * v_rotated;
            if t_top >= 0 && norm(point_top(1:2)) <= R
                point_top = (inverse_rotation_matrix * point_top')';
                current_index = current_index + 1;
                impact_vector(current_index, :) = point_top;
                impact_surface(current_index, :) = 2; % 2 for top cap
            end
        end
    end

    % Trim the preallocated arrays to remove unused space
    impact_vector = impact_vector(1:current_index, :);
    impact_surface = impact_surface(1:current_index, :);
end

function R = get_rotation_matrix(u, v)
    % Calculate the rotation matrix that rotates vector u to vector v
    % u: initial vector (3x1)
    % v: target vector (3x1)

    u = u / norm(u);
    v = v / norm(v);
    cross_uv = cross(u, v);
    dot_uv = dot(u, v);
    
    if norm(cross_uv) < 1e-6
        if dot_uv > 0
            R = eye(3); % No rotation needed
        else
            R = -eye(3); % 180 degree rotation
        end
    else
        skew_cross_uv = [0, -cross_uv(3), cross_uv(2); 
                         cross_uv(3), 0, -cross_uv(1); 
                         -cross_uv(2), cross_uv(1), 0];
        R = eye(3) + skew_cross_uv + skew_cross_uv^2 * ((1 - dot_uv) / norm(cross_uv)^2);
    end
end

function impact_vector_rphiz = convert_impact_data_to_cyl_coords(impact_vector_xyz, impact_surface, vessel_geom)
    % Estimate the number of impact points
    num_points = size(impact_vector_xyz, 1);
    
    % Preallocate output array for cylindrical coordinates [r, phi, z]
    impact_vector_rphiz = zeros(num_points, 3);
    
    % Normalize the axis vector
    axis_cyl = vessel_geom.axis_cyl;
    axis_cyl = axis_cyl / norm(axis_cyl);
    
    % Define the rotation matrix to align the cylinder axis with the z-axis
    z_axis = [0, 0, 1];
    rotation_matrix = get_rotation_matrix(axis_cyl, z_axis);

    % Initialize the index for filling the impact_vector_rphiz array
    current_index = 0;

    % Loop through each impact vector
    for ii = 1:num_points
        if ~isempty(impact_vector_xyz(ii, :))
            % Rotate the impact vector to align with the z-axis
            impact_rotated = (rotation_matrix * impact_vector_xyz(ii, :)')';

            % Initialize r, phi, z for the current impact point
            r = 0;
            phi = 0;
            z = impact_rotated(3);
            
            if impact_surface(ii) == 1  % 1 for cylindrical surface
                r = vessel_geom.R;
                phi = atan2(impact_rotated(2), impact_rotated(1)) * 180 / pi;
            elseif impact_surface(ii) == 2 || impact_surface(ii) == 3  % 2 for top cap, 3 for bottom cap
                r = sqrt(impact_rotated(1)^2 + impact_rotated(2)^2);
                phi = atan2(impact_rotated(2), impact_rotated(1)) * 180 / pi;
            end
            
            % Increment the index and store the result in the preallocated array
            current_index = current_index + 1;
            impact_vector_rphiz(current_index, :) = [r, phi, z];
        end
    end
    
    % Trim the array to remove any unused preallocated space
    impact_vector_rphiz = impact_vector_rphiz(1:current_index, :);
end

function [theta_bins, z_bins, neutral_flux_map] = accumulate_impact_vector(impact_rphiz, vessel_geom, weights, num_theta_bins, num_z_bins)
    % impact_rphiz: Nx3 matrix containing [r, phi, z] coordinates of the impact points
    % vessel_geom: Structure containing cylinder geometry with fields:
    %       - R: Radius of the cylinder
    %       - L: Length of the cylinder
    %       - z0: z-coordinate of the center of the cylinder
    % weights: Nx1 vector of weights corresponding to each impact point
    % num_theta_bins: Number of bins along the circumferential direction (theta)
    % num_z_bins: Number of bins along the height (z)
    
    % Calculate theta and z ranges
    theta_bins = linspace(-pi, pi, num_theta_bins+1); % Discretize theta from -π to π
    z_bottom = vessel_geom.z0 - vessel_geom.L / 2;
    z_top = vessel_geom.z0 + vessel_geom.L / 2;
    z_bins = linspace(z_bottom, z_top, num_z_bins+1); % Discretize z from z_bottom to z_top
    
    % Initialize the neutral_flux_map as a 2D matrix
    neutral_flux_map = zeros(num_z_bins, num_theta_bins);
    
    % Accumulate the weights into the neutral_flux_map
    for i = 1:size(impact_rphiz, 1)
        theta = deg2rad(impact_rphiz(i, 2)); % Convert phi to radians
        z = impact_rphiz(i, 3);
        
        % Find the corresponding bin indices
        [~, theta_idx] = histc(theta, theta_bins);
        [~, z_idx] = histc(z, z_bins);
        
        % Accumulate the weight into the appropriate bin
        if theta_idx > 0 && theta_idx <= num_theta_bins && z_idx > 0 && z_idx <= num_z_bins
            neutral_flux_map(z_idx, theta_idx) = neutral_flux_map(z_idx, theta_idx) + weights(i);
        end
    end
end