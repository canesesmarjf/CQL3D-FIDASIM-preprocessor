function [beam_grid] = calculate_fidasim_beam_grid(inputs)

% Beam grid dimensions in beam_grid coordinates:
xmin = inputs.xmin;
xmax = inputs.xmax;
ymin = inputs.ymin;
ymax = inputs.ymax;
zmin = inputs.zmin;
zmax = inputs.zmax;

% Injected NBI power:
Pinj = inputs.pinj*1e6; % Injected NBI power in [W]

% Define the beam_grid.vertices of the rectangle in 3D space
beam_grid.vertices = ...
[
    xmin, ymin, zmin;
    xmin, ymin, zmax;
    xmin, ymax, zmin;
    xmin, ymax, zmax;
    xmax, ymin, zmin;
    xmax, ymin, zmax;
    xmax, ymax, zmin;
    xmax, ymax, zmax
];

% Convert to a collection of 8 column vectors:
beam_grid.vertices = beam_grid.vertices';

% Define the beam_grid.faces of the rectangle using the beam_grid.vertices
beam_grid.faces = ...
[
    1, 2, 4, 3; % Side face (xmin)
    5, 6, 8, 7; % Side face (xmax)
    1, 2, 6, 5; % Side face (ymin)
    3, 4, 8, 7; % Side face (ymax)
    1, 3, 7, 5; % Bottom face
    2, 4, 8, 6  % Top face
];
beam_grid.faces = beam_grid.faces';

% Beam grid orientation:
alpha = inputs.alpha;
beta  = inputs.beta;
ca = cos(alpha);
cb = cos(beta);
sa = sin(alpha);
sb = sin(beta);

% Rotation matrix also referred to as basis in FIDASIM in subroutine
% tb_zxy:
% It is used to create the beam grid coordinate system [XYZ] by rotating the
% machine coorindate system [UVW] as follows:
% [XYZ] = [UVW]*basis

beam_grid.basis = [ca*cb, -sa, ca*sb; ...
                   cb*sa, +ca, sa*sb; ...
                   -sb  ,  0 ,    cb];

% Column vector for origin of beam grid in UVW:
beam_grid.origin_uvw = inputs.origin';

% Compute the beam grid vertices in UVW machine coordinate system:
for ii = 1:size(beam_grid.vertices,2)
    beam_grid.vertices_uvw(:,ii) = beam_grid.origin_uvw + beam_grid.basis*beam_grid.vertices(:,ii);
end

% Compute a beam grid in XYZ:
beam_grid.nx = inputs.nx;
beam_grid.xmin = inputs.xmin;
beam_grid.xmax = inputs.xmax;
beam_grid.ny = inputs.ny;
beam_grid.ymin = inputs.ymin;
beam_grid.ymax = inputs.ymax;
beam_grid.nz = inputs.nz;
beam_grid.zmin = inputs.zmin;
beam_grid.zmax = inputs.zmax;
beam_grid.dx = (inputs.xmax - inputs.xmin)/inputs.nx;
beam_grid.dy = (inputs.ymax - inputs.ymin)/inputs.ny;
beam_grid.dz = (inputs.zmax - inputs.zmin)/inputs.nz;

ii = 1:1:beam_grid.nx;
beam_grid.x = beam_grid.xmin + (ii-0.5)*beam_grid.dx;

ii = 1:1:beam_grid.ny;
beam_grid.y = beam_grid.ymin + (ii-0.5)*beam_grid.dy;

ii = 1:1:beam_grid.nz;
beam_grid.z = beam_grid.zmin + (ii-0.5)*beam_grid.dz;

% Package grid's dimensions into vector:
beam_grid.ng = [beam_grid.nx,beam_grid.ny, beam_grid.nz];

% Define the beam grid's spatial extents:
beam_grid.length(1) = beam_grid.xmax - beam_grid.xmin;
beam_grid.length(2) = beam_grid.ymax - beam_grid.ymin;
beam_grid.length(3) = beam_grid.zmax - beam_grid.zmin;

end