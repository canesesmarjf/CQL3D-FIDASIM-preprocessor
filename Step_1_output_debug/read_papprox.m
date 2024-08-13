% reads data stored from a FIDASIM debug session in forge
% We are looking at the structure of ppaprox used in the DCX subroutine to
% initiate the new neutrals

clear all
close all

input_dir  = "./";

% Create beam grid:
% =========================================================================
file_path = input_dir + "WHAM_example_debug_inputs.dat";
input = read_namelist(file_path);

% Extract data:
xmin = input.xmin;
xmax = input.xmax;
nx   = input.nx;
ymin = input.ymin;
ymax = input.ymax;
ny   = input.ny;
zmin = input.zmin;
zmax = input.zmax;
nz   = input.nz;

% Beam grid increments:
dX = (xmax - xmin)/nx;
dY = (ymax - ymin)/ny;
dZ = (zmax - zmin)/nz;

% Beam grid centers:
for ii = 1:nx
    Xc(ii) = xmin + (ii - 0.5)*dX;
end

for ii = 1:ny
    Yc(ii) = ymin + (ii - 0.5)*dY;
end

for ii = 1:nz
    Zc(ii) = zmin + (ii - 0.5)*dZ;
end

% Beam grid edges:
if 1
    X = [Xc - 0.5*dX,xmax];
    Y = [Yc - 0.5*dY,ymax];
    Z = [Zc - 0.5*dZ,zmax];
else
    X = linspace(xmin,xmax,nx+1);
    Y = linspace(ymin,ymax,ny+1);
    Z = linspace(zmin,zmax,nz+1);
end

% Assemble papprox:
% =========================================================================
papprox = storeDataIn3DArray('papprox');
nlaunch0 = storeDataIn3DArray('nlaunch0');

% Plot papprox profiles:
% =========================================================================
figure
hold on
prof_XZ = permute(sum(papprox,2),[1 3 2]);
hp = pcolor(Zc,Xc,log(prof_XZ));
[ho,hc] = contourf(Zc,Xc,log(prof_XZ));

axis('image')
colormap(flipud(hot))

% Add the grid edges::
[xx,zz] = meshgrid(X,Z);
plot(zz ,xx ,'k-','LineWidth',0.25)
plot(zz',xx','k-','LineWidth',0.25)

% Add the grid centers::
[xx,zz] = meshgrid(Xc,Zc);
plot(zz ,xx ,'r.','LineWidth',0.25)
plot(zz',xx','r.','LineWidth',0.25)


% Plot nlaunch0 profiles:
% =========================================================================
figure
hold on
prof_XZ = permute(sum(nlaunch0,2),[1 3 2]);
[ho,hc] = contourf(Zc,Xc,log(prof_XZ));

axis('image')
colormap(flipud(hot))

% Add the grid edges::
[xx,zz] = meshgrid(X,Z);
plot(zz ,xx ,'k-','LineWidth',0.25)
plot(zz',xx','k-','LineWidth',0.25)

% Add the grid centers::
[xx,zz] = meshgrid(Xc,Zc);
plot(zz ,xx ,'r.','LineWidth',0.25)
plot(zz',xx','r.','LineWidth',0.25)
title('nlaunch0')
colorbar

%% functions:
function papprox = storeDataIn3DArray(filename)
    % Read the data from the file
    data = readtable(filename);

    % Extract the indices and values
    i = table2array(data(:, 1));
    j = table2array(data(:, 2));
    k = table2array(data(:, 3));
    values = table2array(data(:, 4));

    % Determine the size of the 3D array
    max_i = max(i);
    max_j = max(j);
    max_k = max(k);

    % Initialize the 3D array
    papprox = zeros(max_i, max_j, max_k);

    % Store the values in the 3D array
    for idx = 1:length(values)
        papprox(i(idx), j(idx), k(idx)) = values(idx);
    end
end

function data = read_namelist(file_path)
    % Initialize an empty structure
    data = struct();

    % Open the file for reading
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Cannot open the file.');
    end

    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);

        % Skip comments and empty lines
        if isempty(line) || startsWith(line, '!!') || startsWith(line, '&') || startsWith(line, '/')
            continue;
        end

        % Parse the line
        [key, value] = parse_line(line);

        % Assign the value to the structure
        if ~isempty(key)
            % Handle array elements
            array_expr = regexp(key, '(.*)\((\d+)\)', 'tokens');
            if ~isempty(array_expr)
                base_key = array_expr{1}{1};
                index = str2double(array_expr{1}{2});
                if ~isfield(data, base_key)
                    data.(base_key) = [];
                end
                data.(base_key)(index) = value;
            else
                data.(key) = value;
            end
        end
    end

    % Close the file
    fclose(fid);
end

function [key, value] = parse_line(line)
    % Remove comments from the line
    line = regexprep(line, '!!.*', '');

    % Split the line into key and value
    parts = strsplit(line, '=');
    if length(parts) ~= 2
        key = '';
        value = '';
        return;
    end

    key = strtrim(parts{1});
    value_str = strtrim(parts{2});

    % Convert the value string to a MATLAB data type
    if startsWith(value_str, '''') && endsWith(value_str, '''')
        % String value
        value = strrep(value_str, '''', '');
    elseif contains(value_str, ',')
        % Array value
        value = str2num(value_str); %#ok<ST2NM>
    else
        % Single numeric value
        value = str2double(value_str);
        if isnan(value)
            value = value_str; % Keep as string if not a number
        end
    end
end