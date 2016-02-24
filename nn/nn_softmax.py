#coding:utf-8

import os
import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import data_loader
from activator import sigmoid, softmax

def np_extend(x, val = 0):
    '''在矩阵x的基础上, 补上一列0或1
    '''
    assert val == 0 or val == 1
    y = np.empty((x.shape[0], x.shape[1] + 1))
    y[:, 1:] = x
    y[:, 0] = val
    return y

def fp(theta, ninput, nhidden, noutput, Lambda, X, y):
    '''正向传播求得损失函数
    '''
    theta1 = np.reshape(theta[0:nhidden*(ninput+1)], [nhidden, ninput + 1])
    theta2 = np.reshape(theta[nhidden*(ninput+1):],  [noutput, nhidden + 1])

    m = X.shape[0]

    a1 = np_extend(X, 1)
    z2 = np.dot(a1, theta1.T)
    a2 = np_extend(sigmoid(z2), 1)
    z3 = np.dot(a2, theta2.T)

    ## sigmoid
    # a3 = sigmoid(z3)

    # yTmp = np.eye(noutput)
    # yy = yTmp[y][:]

    # J = np.sum(np.sum(-yy*np.log(a3)-(1-yy)*np.log(1-a3))) / m;
    # J += 0.5*Lambda/m * (np.sum(np.sum(theta1[:,1:]*theta1[:,1:])) + \
                         # np.sum(np.sum(theta2[:,1:]*theta2[:,1:])));
    
    # softmax
    a3 = softmax(z3)
    J = - np.sum(np.log(a3[range(m), y]));
    J += 0.5*Lambda/m* (np.sum(np.sum(theta1[:,1:]*theta1[:,1:])) + \
                         np.sum(np.sum(theta2[:,1:]*theta2[:,1:])));
    
    return J


def bp(theta, ninput, nhidden, noutput, Lambda, X, y):
    '''反向传播, 求得theta的梯度, 这里有很多计算是和fp重复的, 原因在于迭代函数
    fmin_cg的参数格式要求, 重复的程度很高, 很影响效率
    '''
    theta1 = np.reshape(theta[0:nhidden*(ninput+1)], [nhidden, ninput + 1])
    theta2 = np.reshape(theta[nhidden*(ninput+1):],  [noutput, nhidden + 1])

    m = X.shape[0]

    a1 = np_extend(X, 1)
    z2 = np.dot(a1, theta1.T)
    a2 = np_extend(sigmoid(z2), 1)
    z3 = np.dot(a2, theta2.T)

    ## sigmoid
    # a3 = sigmoid(z3)

    # softmax
    a3 = softmax(z3)

    yTmp = np.eye(noutput)
    yy = yTmp[y][:]

    delta3 = a3 - yy
    delta2 = np.dot(delta3, theta2[:, 1:]) * a2[:, 1:] * (1-a2[:, 1:])

    theta1_g = np_extend(Lambda / m * theta1[:, 1:])
    theta2_g = np_extend(Lambda / m * theta2[:, 1:])
    theta1_g += 1.0 / m * np.dot(delta2.T, a1)
    theta2_g += 1.0 / m * np.dot(delta3.T, a2)

    grad = np.empty(theta.shape)
    grad[0:nhidden*(ninput+1)] = np.reshape(theta1_g, nhidden * (ninput + 1))
    grad[nhidden*(ninput+1):] = np.reshape(theta2_g, noutput * (nhidden + 1))

    return grad


def predict(theta, X, ninput, nhidden, noutput):
    theta1 = np.reshape(theta[0:nhidden*(ninput+1)], [nhidden, ninput + 1])
    theta2 = np.reshape(theta[nhidden*(ninput+1):],  [noutput, nhidden + 1])

    h1 = sigmoid(np.dot(np_extend(X, 1), theta1.T))
    h2 = sigmoid(np.dot(np_extend(h1, 1), theta2.T))    
    return np.argmax(h2, axis = 1)

def print_each_iter(theta):
    print 'this iteration runs so long time...'


def train(X, y, ninput, nhidden, noutput, Lambda):
    theta = 0.1 * np.random.randn(nhidden * (ninput+1) + noutput * (nhidden+1))

    res = fmin_cg(fp, theta,
        fprime = bp,
        args = (ninput, nhidden, noutput, Lambda, X, y),  
        maxiter = 50,
        callback = print_each_iter,
        full_output = True)
    
    print 'Iteration finished.'
    print '\toptimal cost: %2f' % res[1]
    print '\t"fp" called: %d times' % res[2]
    print '\t"bp" called: %d times' % res[3]

    return res[0]


################################################################################
if __name__ == "__main__":
    print 'Loading data...'
    train_X, train_y, test_X, test_y = data_loader.load_all()
    print 'Data Loaded.\n\n'

    ninput = train_X.shape[1]
    nhidden = 20
    noutput = 10
    Lambda = 0;

    print 'Training Neural Network Model....'
    theta = train(test_X, test_y, ninput, nhidden, noutput, Lambda)
    y_p = predict(theta, train_X, ninput, nhidden, noutput)
    print 'Training Accuracy: %.2f ---(%d/%d)' % (np.mean(y_p == train_y), 
        np.sum(y_p == train_y), train_y.shape[0])

    print 'Training process done.\n\n'

    print 'Testing...'
    y_p = predict(theta, test_X, ninput, nhidden, noutput)
    print 'Testing Accuracy: %.2f ---(%d/%d)' % (np.mean(y_p == test_y), 
        np.sum(y_p == test_y), test_y.shape[0])
    print 'Testing done.\n\n'