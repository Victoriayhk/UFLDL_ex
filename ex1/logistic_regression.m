function [f,g] = logistic_regression(theta, X, y )
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  [n, m] = size(X);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

h = sigmoid(theta' * X);
f = -sum(y .* log(h) + (1 - y) .* log(1 - h));
g = sum(bsxfun(@times, X, h - y), 2);
% g = sum(X .* repmat(h - y, n, 1), 2);
