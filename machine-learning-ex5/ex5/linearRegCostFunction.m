function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
sum =0;
O = ones(m,1);

h = X*theta;
J = h-y;
P=J;
for i=1:m,
  sum = sum + J(i)^2;
end;

J = (1/(2*m))*sum;

add=0;
new = theta .* theta;
for j=2:n,
  add = add + new(j);
end;
J = J + (lambda/(2*m))*add;

L = P'*X;
grad(1) = (1/m)*L(1);

for l=2:n,
  grad(l) = (1/m)*L(l) + (lambda/m)*theta(l);
  










% =========================================================================

grad = grad(:);

end
