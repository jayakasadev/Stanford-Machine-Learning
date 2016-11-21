function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% =========================================================================

% C_arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300; 1000; 3000];
% sigma_arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300; 1000; 3000];

C_arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10];
sigma_arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10];

errors = zeros(length(C_arr), length(sigma_arr));

for i = 1 : length(C_arr)
    C = C_arr(i);
    
    for a = 1 : length(sigma_arr)
        sigma = sigma_arr(a);
        
        % training the mdoel based on the chosen sigma and C value
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % visualizeBoundary(X, y, model);
        
        predictions = svmPredict(model, Xval);
        
        errors(i,a) = mean(double(predictions ~= yval));
    end
end

% getting the lowest error
error = min(min(errors));
% getting the location of that error
[i , j] = find(errors == error);
% getting the corresponding C and sigma
C = C_arr(i);
sigma = sigma_arr(j);
    
end
