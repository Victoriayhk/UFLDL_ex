function [cost, grad, p] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

t = exp(theta * data);
p = bsxfun(@rdivide, t, sum(t));

cost = (-1/m) .* sum(sum(groundTruth.*log(p)));
cost = cost + (lambda/2).*sum(sum(theta.^2));
thetagrad = (-1/m) .* ((groundTruth-p)*data') + (lambda*theta);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];

end

