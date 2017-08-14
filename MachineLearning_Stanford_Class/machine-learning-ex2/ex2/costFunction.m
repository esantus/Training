function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% m already calculated above, but equivalent function below
% m = size(X, 1)

% sigmoid of each sample in X multiplied by theta
h0 = sigmoid(X * theta);

% error calculation
Error = (sum(-y .* log(h0)) - sum((1 - y) .* log(1 - h0)));
% Error = (transpose(-y) * log(h0)) - ((1 - transpose(y)) * log(1 - h0));


% cost calculation
J = (1/m) * Error;

% gradient of the cost
grad = (1/m) * transpose(X) * (h0 - y);




% =============================================================

end
