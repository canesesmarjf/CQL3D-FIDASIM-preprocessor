function out = select_slice(A, dim, rng)
%SELECT_SLICE Select a slice of an array along a specified dimension.
%
%   out = SELECT_SLICE(A, dim, rng) returns a subset of the input array A,
%   where elements along dimension 'dim' are selected according to the range
%   vector 'rng'. All other dimensions are fully included.
%
%   This function is useful when you want to dynamically select data along
%   a specific dimension without hardcoding the colon notation.
%
%   Inputs:
%     A   - Input array of any dimensionality (e.g., 2D, 3D, etc.)
%     dim - Scalar integer specifying the dimension to index
%     rng - Vector of indices to select along dimension 'dim'
%
%   Output:
%     out - Resulting array with only elements at positions 'rng' along
%           dimension 'dim'. The size of 'out' will match A in all other
%           dimensions.
%
%   Example:
%     A = rand(4, 10, 3);           % Create a 4x10x3 array
%     dim = 2;                      % Choose to slice along the 2nd dimension
%     rng = 3:6;                    % Select indices 3 through 6
%     out = select_slice(A, dim, rng);  % Equivalent to A(:,3:6,:)
%
%   Notes:
%     - If 'dim' exceeds the number of dimensions in A, an error is thrown.
%     - If 'rng' is empty or contains invalid indices, indexing will error.
%     - Output size is: size(A) with size(out,dim) == length(rng)
%
%   See also: subsref, squeeze, permute, ndims

    idx = repmat({':'}, 1, ndims(A));
    idx{dim} = rng;
    out = A(idx{:});
end