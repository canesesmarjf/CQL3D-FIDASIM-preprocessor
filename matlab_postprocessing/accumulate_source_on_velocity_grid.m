function [N_weighted, Xcenters, Ycenters] = accumulate_source_on_velocity_grid(V1, V2, W, nbinsX, nbinsY)
    % compute_weighted_velocity_distribution - Compute the weighted distribution
    % of velocities on a 2D grid.
    %
    % Syntax:  [N_weighted, Xcenters, Ycenters] = compute_weighted_velocity_distribution(V1, V2, W, nbinsX, nbinsY)
    %
    % Inputs:
    %   V1 - 1D array of parallel velocities
    %   V2 - 1D array of perpendicular velocities
    %   W  - 1D array of weights corresponding to V1 and V2
    %   nbinsX - Number of bins for V1 (x-axis)
    %   nbinsY - Number of bins for V2 (y-axis)
    %
    % Outputs:
    %   N_weighted - 2D array containing the weighted distribution
    %   Xcenters - Centers of the bins for V1
    %   Ycenters - Centers of the bins for V2
    
    % Determine the range of V1 and V2 with an extra 20% space
    rangeV1 = max(V1) - min(V1);
    rangeV2 = max(V2) - min(V2);
    
    minV1 = min(V1) - 0.1 * rangeV1;
    maxV1 = max(V1) + 0.1 * rangeV1;
    minV2 = min(V2) - 0.1 * rangeV2;
    maxV2 = max(V2) + 0.1 * rangeV2;
    
    % Create a uniform grid spanning the adjusted range of V1 and V2
    Xedges = linspace(minV1, maxV1, nbinsX + 1);
    Yedges = linspace(minV2, maxV2, nbinsY + 1);
    
    % Calculate the centers of each bin
    Xcenters = (Xedges(1:end-1) + Xedges(2:end)) / 2;
    Ycenters = (Yedges(1:end-1) + Yedges(2:end)) / 2;
    
    % Initialize the weighted distribution grid
    N_weighted = zeros(nbinsX, nbinsY);
    
    % Loop over each particle and accumulate the weight in the appropriate cell
    for i = 1:length(V1)
        % Find the indices of the bin that V1(i) and V2(i) fall into
        [~, binX] = histc(V1(i), Xedges);
        [~, binY] = histc(V2(i), Yedges);
        
        % Ensure that the bin indices are within valid ranges
        if binX > 0 && binX <= nbinsX && binY > 0 && binY <= nbinsY
            N_weighted(binX, binY) = N_weighted(binX, binY) + W(i)*V2(i);
        end
    end
end