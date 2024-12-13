% function [N_weighted, Xcenters, Ycenters] = accumulate_source_on_velocity_grid(V1, V2, W, nbinsX, nbinsY)
%     % compute_weighted_velocity_distribution - Compute the weighted distribution
%     % of velocities on a 2D grid.
%     %
%     % Syntax:  [N_weighted, Xcenters, Ycenters] = compute_weighted_velocity_distribution(V1, V2, W, nbinsX, nbinsY)
%     %
%     % Inputs:
%     %   V1 - 1D array of parallel velocities
%     %   V2 - 1D array of perpendicular velocities
%     %   W  - 1D array of weights corresponding to V1 and V2
%     %   nbinsX - Number of bins for V1 (x-axis)
%     %   nbinsY - Number of bins for V2 (y-axis)
%     %
%     % Outputs:
%     %   N_weighted - 2D array containing the weighted distribution
%     %   Xcenters - Centers of the bins for V1
%     %   Ycenters - Centers of the bins for V2
%     
%     % Determine the range of V1 and V2 with an extra 20% space
%     rangeV1 = max(V1) - min(V1);
%     rangeV2 = max(V2) - min(V2);
%     
%     minV1 = min(V1) - 0.1 * rangeV1;
%     maxV1 = max(V1) + 0.1 * rangeV1;
%     minV2 = min(V2) - 0.1 * rangeV2;
%     maxV2 = max(V2) + 0.1 * rangeV2;
%     
%     % Create a uniform grid spanning the adjusted range of V1 and V2
%     Xedges = linspace(minV1, maxV1, nbinsX + 1);
%     Yedges = linspace(minV2, maxV2, nbinsY + 1);
%     
%     % Calculate the centers of each bin
%     Xcenters = (Xedges(1:end-1) + Xedges(2:end)) / 2;
%     Ycenters = (Yedges(1:end-1) + Yedges(2:end)) / 2;
%     
%     % Initialize the weighted distribution grid
%     N_weighted = zeros(nbinsX, nbinsY);
%     
%     % Loop over each particle and accumulate the weight in the appropriate cell
%     for i = 1:length(V1)
%         % Find the indices of the bin that V1(i) and V2(i) fall into
%         [~, binX] = histc(V1(i), Xedges);
%         [~, binY] = histc(V2(i), Yedges);
%         
%         % Ensure that the bin indices are within valid ranges
%         if binX > 0 && binX <= nbinsX && binY > 0 && binY <= nbinsY
%             N_weighted(binX, binY) = N_weighted(binX, binY) + W(i)*V2(i);
%         end
%     end
% end
function [N_weighted, Xcenters, Ycenters] = accumulate_source_on_velocity_grid(Vpar, Vperp, W, nbinsX, nbinsY)
    % accumulate_source_on_velocity_grid - Compute the weighted distribution
    % of velocities on a 2D grid, including the Jacobian for cylindrical coordinates.
    %
    % Syntax:  [N_weighted, Xcenters, Ycenters] = accumulate_source_on_velocity_grid(Vpar, Vperp, W, nbinsX, nbinsY)
    %
    % Inputs:
    %   Vpar - 1D array of parallel velocities (V_parallel)
    %   Vperp - 1D array of perpendicular velocities (V_perpendicular)
    %   W - 1D array of weights corresponding to particles
    %   nbinsX - Number of bins for Vpar (x-axis)
    %   nbinsY - Number of bins for Vperp (y-axis)
    %
    % Outputs:
    %   N_weighted - 2D array containing the weighted distribution with Jacobian
    %   Xcenters - Centers of the bins for Vpar
    %   Ycenters - Centers of the bins for Vperp

    % Determine the range of Vpar and Vperp with an extra 20% padding
    rangeVpar = max(Vpar) - min(Vpar);
    rangeVperp = max(Vperp) - min(Vperp);
    
    minVpar = min(Vpar) - 0.1 * rangeVpar;
    maxVpar = max(Vpar) + 0.1 * rangeVpar;
    minVperp = min(Vperp) - 0.1 * rangeVperp;
    maxVperp = max(Vperp) + 0.1 * rangeVperp;
    
    % Create a uniform grid spanning the adjusted range of Vpar and Vperp
    Xedges = linspace(minVpar, maxVpar, nbinsX + 1);
    Yedges = linspace(minVperp, maxVperp, nbinsY + 1);
    
    % Calculate the centers of each bin
    Xcenters = (Xedges(1:end-1) + Xedges(2:end)) / 2;
    Ycenters = (Yedges(1:end-1) + Yedges(2:end)) / 2;
    
    % Initialize the weighted distribution grid
    N_weighted = zeros(nbinsX, nbinsY);
    
    % Loop over each particle and accumulate the weight in the appropriate cell
    for i = 1:length(Vpar)
        % Find the indices of the bin that Vpar(i) and Vperp(i) fall into
        binX = find(Xedges <= Vpar(i), 1, 'last');
        binY = find(Yedges <= Vperp(i), 1, 'last');
        
        % Ensure that the bin indices are within valid ranges
        if binX > 0 && binX <= nbinsX && binY > 0 && binY <= nbinsY
            % Apply Jacobian and accumulate weight
            N_weighted(binX, binY) = N_weighted(binX, binY) + W(i) * abs(Vperp(i));
        end
    end
end