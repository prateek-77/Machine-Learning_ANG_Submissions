function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h = X*theta;
% You need to return the following variables correctly 
J = 0;
E=0;
grad = zeros(size(theta));
n = length(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
Q = (-y' * log(sigmoid(h)));
W = (ones(m,1) - y)' * log(ones(m,1) - sigmoid(h));
for i=2:n,
  E = E + theta(i)^2;
end;
E = (lambda/(2*m))*E;
J =  (1/m)*(Q-W) + E;

grad = (1/m)*X'*(sigmoid(h)-y)

for k=2:n,
  grad(k) = grad(k) + (lambda/m)*theta(k);
end;

grad(1) = (1/m)*X(:,1)'*(sigmoid(h)-y)











% =============================================================

end
