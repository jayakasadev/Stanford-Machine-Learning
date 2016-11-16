function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% =========================================================================

% had to use theta(2:end) because the theta array is a vector

% size(X)
% size(theta)
% (12 x 1) = (12 x 2) (2 x 1)
H = X * theta;
% H
% (12 x 1)
Difference = (H - y);
% (1 x 1) = sum(12 x 1) + sum(0)
J = (1/(2 * m)) * sum(Difference.^2) + (lambda / (2 * m)) * sum(theta(2:end).^2);
% J
% size(J)

% grad = grad(:);

% (2 x 1) = (2 x 12) * (12 x 1)
grad = (1 / m) * X' * Difference;

%(1 x 1) = (1 x 1) + (1 x 1)
grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);
% grad
% size(grad)

end
