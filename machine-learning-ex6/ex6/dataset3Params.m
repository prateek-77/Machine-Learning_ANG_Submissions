function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
n= size(Xval);
%function [model] = svmTrain(X, Y, C, kernelFunction, ...
%                           tol, max_passes)
%function pred = svmPredict(model, X)
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
%P = zeros(64,1);
L = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]';
error = 99999999999;
C1 = 099999999999999;
sigma1 = 99999999999999;

for i=1:8,
	for j=1:8,
		model = svmTrain(X,y, L(i), @(x1, x2) gaussianKernel(x1, x2, L(j)));
		predictions = svmPredict(model, Xval);
		error1 = mean(double(predictions ~= yval));
		if error1 < error
			error = error1;
			C1 = L(i);
			sigma1 = L(j);
		end;
	end;
end;

C = C1;
sigma = sigma1;


% =========================================================================

end
