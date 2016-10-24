function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% (118*1)
h = sigmoid(X * theta);

%regularization values
rtheta = theta;

% (28*1)
rtheta(1) = 0;
rcost = (lambda / (2 * m))  * sum(rtheta.^2);

% (28*1)
rgrad = (lambda / m) * rtheta;

% calculating the cost and gradient value with regularization
J = (1 / m) * sum((- y .* log(h)) - ((1 - y) .* log(1 - h))) + rcost;

% (1*28) + (1*28)
% the solution was to get the transpose of rgrad and change rtheta so that
% it sets the first element to 0 instead of trimming or resizing the array
grad = (1 / m) * sum((h - y) .* X) + rgrad';




% =============================================================

end
