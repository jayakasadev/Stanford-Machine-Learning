function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% =============================================================

% getting the number of examples from the first dimension of X
m = size(X, 1);

% centroids = [ 2.428301 3.157924; 5.813503 2.633656; 7.119387 3.616684 ]

for i = 1:m
    % training example i
    x = X(i, :);
    
    % the below line is good for finding exact matches only
    % [~, a, b] = intersect(centroids, x, 'rows');
    
    % [c index] = min(abs(centroids-X(1,:)))
    % [val, index] = min(abs(centroids - x).^2);
    [val, index] = min(pdist2(centroids, x, 'euclidean'));
    idx(i) = index(1);
end

end

