function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight 
% matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta1_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
%% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% ()

% this code here will expand the y array which is (5000 x 1) into Y which
% is (5000 * 10)

% BUILDING A LOGICAL ARRAY
% (10x10)
I = eye(num_labels);
% (5000 x 10)
Y = zeros(m, num_labels);
for i=1:m
    Y(i, :)= I(y(i), :);
end

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
h = sigmoid(z3);

% If you use just the regular y here, the matrix dimensions do not match,
% and you will end up with an incorrect answer. The key is to make sure
% that your y array has the same dimensions as your h array. 
% The directions said to recode the labels in the y array so that it only
% contains 0s and 1s. This is for the purpose of training a neural network.

% So, you are turning the y vector into a 2d vector where you have a
% correct output vector for each training sample. 
% for  i = 1 : m
% y(i) = 4      -> [0 0 0 1 0 0 0 0 0 0]
% y(i+1) = 5    -> [0 0 0 0 1 0 0 0 0 0]
% etc.

% This are requires further testing as to why it works.
J = (1 / m) * sum(sum(-Y .* log(h) - ((1 - Y) .* log(1-h))));

% Getting the cost of each Theta array and summing it all up into one value
% Regularization value
jreg = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

% This bit of code does not work because the dimensions of Theta1 and
% Theta2 do not match.
% jreg = (lambda / (2 * m)) * sum(sum(Theta1(:, 2:end).^2 + Theta2(:, 2:end).^2));

J = J + jreg;

%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% subtract the calculated values from the expected values.
% must use the Y array in order for the dimensions of the 2 arrays to
% match

% directions say to do this in a for loop in order to implement a learning
% mechanism. POSSIBLE SOURCE OF BUG. 
% Going to vectorize the whole process

% (5000x10)
delta3 = h - Y;
% the matrix dimensions do not match here
% delta2 = Theta2' * delta3 .* sigmoidGradient(z2);

% (5000x26)
delta2 = delta3 * Theta2;
% (5000x25) = (5000x25) .* (5000x25)
delta2 = delta2(:, 2:end) .* sigmoidGradient(z2);

% (10x26) = (10x5000) * (5000x26)
Delta2 = delta3' * a2;
% (25*401) = (25x5000) * (5000x401)
Delta = delta2' * a1;

% (25*401)
Theta1_grad = (1 / m) * Delta;
% (10*26)
Theta2_grad = (1 / m) * Delta2;

%% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% (25x401)
Theta1_grad_reg = (lambda / m) * Theta1;
% (10x26)
Theta2_grad_reg = (lambda / m) * Theta2;

% (25*401) = (25*401) + (25*401)
% setting the first column to zero
Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad_reg, 1), 1) Theta1_grad_reg(:, 2:end)];
% Theta1_grad = Theta1_grad + Theta1_grad_reg;

% (10x26) = (10x26) + (10x26)
% setting the first column to zero
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad_reg, 1), 1) Theta2_grad_reg(:, 2:end)];
% Theta2_grad = Theta2_grad + Theta2_grad_reg;

%% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
