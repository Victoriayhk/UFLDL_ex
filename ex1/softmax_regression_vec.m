function [f, g, p] = softmax_regression_vec(theta, X, y)
%
% Arguments:
%   theta - A vector containing the parameter values to optimize.
%       In minFunc, theta is reshaped to a long vector.  So we need to
%       resize it to an n-by-(num_classes-1) matrix.
%       Recall that we assume theta(:,num_classes) = 0.
%
%   X - The examples stored in a matrix.  
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example.  y(j) is the j'th example's label.
%
m=size(X,2);
n=size(X,1);

% theta is a vector;  need to reshape to n x num_classes.
theta=reshape(theta, n, []);
num_classes=size(theta,2)+1;

% initialize objective value and gradient.
% f = 0;
% g = zeros(size(theta));

%
% TODO:  Compute the softmax objective function and gradient using vectorized code.
%        Store the objective function value in 'f', and the gradient in 'g'.
%        Before returning g, make sure you form it back into a vector with g=g(:);
%
%%% YOUR CODE HERE %%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 关于theta为什么可以只训练前K-1个，见
% http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
% 可以稍微提一下的是，因为softmax函数的特性，K个theta同时减去一个相同的向量，
% 其最后算得的假说函数值相同（exp指数部分加减提出为乘法形式再约去），故可约定
% K个theta一起减去Theta_K即可。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta = [theta, zeros(size(theta, 1), 1)];  % 将theta扩展为n*K维以便后续操作
t = exp(theta' * X);
p = bsxfun(@mrdivide, t, sum(t));           % 概率p

pyi = sub2ind(size(p), y, 1:size(p, 2));
f = -sum(log(p(pyi)));                      % 损失函数

g = X * p';
for i = 1 : m
    g(:, y(i)) = g(:, y(i)) - X(:, i);
end
g = g(:, 1:end - 1);                        % 损失函数对theta的偏导，注意维度
g = g(:); % make gradient a vector for minFunc

