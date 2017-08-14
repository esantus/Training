function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h0 = sigmoid(X * theta);

% initializing a copy of theta with first value = 0, as we do not want to regularise theta0
theta1 = [0 ; theta(2:size(theta), :)];

% square of theta (with theta0=0)
sqr = theta1' * theta1;

% regularizers
cost_regularizer = (lambda / (2 * m)) * sqr;
grad_regularizer = (lambda / m) * theta1;

% Error
Error = ((-y)' * log(h0)) - ((1 - y)' * log(1 - h0));

% regularised cost
J = ((1/m) * Error) + cost_regularizer;

% regularised gradient
grad = ((1/m) * (X' * (h0 - y))) + grad_regularizer;

% =============================================================

end
