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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);       
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
O = ones(m,1);
z1 = [O X];
z2 = z1 * Theta1';
h1 = sigmoid(z2);
h1 = [O h1];
z3 = h1 * Theta2';
h2 = sigmoid(z3);


y1 = zeros(size(y,1), 10);
y1(sub2ind(size(y1), (1:size(y1,1))', y)) = 1;

%J = -y1'*log(h2) - ((ones(size(y1)))' - y1')*(log(ones(size(h2)) - h2))
%J = J/m;
%J = sum(J(:))
J=0;
for i = 1:m,
	for k = 1:num_labels,
		J = J + -y1(i,k)*log(h2(i,k))-(1-y1(i,k))*log(1-h2(i,k));
    end;
end;
J= J/m;
sum1=0;
for p=1:(hidden_layer_size),
  for q=2:(n+1),
    sum1 = sum1 + Theta1(p,q)^2;
  end;
end;
    
for c=1:(num_labels),
  for v=2:(hidden_layer_size+1),
    sum1 = sum1 + Theta2(c,v)^2;
  end;
end;
  
J = J + (lambda/(2*m))*sum1;
D1 = 0;
D2 = 0;
D3 = 0;

for t = 1:m,
 
  O = ones(m,1);
  z1 = [O X];
  z2 = z1 * Theta1';
  h1 = sigmoid(z2);
  h1 = [O h1];
  z3 = h1 * Theta2';
  h2 = sigmoid(z3);
  d3 = (h2(t,:) - y1(t,:))';
  Z2 = z2(t,:);
  Z2 = [0 Z2]
  d2 = ((Theta2)'*d3) .* sigmoidGradient(Z2)';
  d2 = d2(2:end);
  D2 = D2 + d3'* h2'(t);
  D2 = D2/m;
  
 end;


grad = D2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
