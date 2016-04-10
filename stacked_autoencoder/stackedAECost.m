function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% 1. ǰ������
acti = cell(1, numel(stack) + 1);
acti{1} = data;
for d = 1:numel(stack)
    acti{d+1} = sigmoid(bsxfun(@plus, stack{d}.w * acti{d}, stack{d}.b));
end

% 2. ��������ݶ�(softmax)����ʧ����
[cost, softmaxThetaGrad, softmaxActi] = softmaxCost(softmaxTheta, ...
                                            numClasses, hiddenSize, ...
                                            lambda, acti{end}, labels);

% Note: When adding the weight decay term to the cost, you should regularize
% only the softmax weights (do not regularize the weights that compute the
% hidden layer activations) ����Ϊԭ�Ĺʴ˴�ע�͵�, ����������lambda*w����
% for d = 1 : numel(stack)
%     cost = cost + (0.5*lambda) * sum(sum(stack{d}.w .^ 2));
% end

% 3. ������

delta = cell(1, numel(stack) + 1);
softmaxDelta = softmaxActi - groundTruth;
delta{end} = (softmaxTheta' * softmaxDelta) .* (acti{end} .* (1-acti{end}));

for l = numel(delta)-1:-1:2
    delta{l} = (stack{l}.w' * delta{l+1}) .* (acti{l} .* (1 - acti{l}));
end

for l = 1:numel(stack)
    stackgrad{l}.w = (1/M) * (delta{l+1} * acti{l}');% + lambda * stack{l}.w;
    stackgrad{l}.b = (1/M) * sum(delta{l+1}, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end