import os
import struct
from array import array
import numpy as np

path = 'data/'
test_img_fname = 't10k-images-idx3-ubyte'
test_lbl_fname = 't10k-labels-idx1-ubyte'
train_img_fname = 'train-images-idx3-ubyte'
train_lbl_fname = 'train-labels-idx1-ubyte'

test_images = []
test_labels = []
train_images = []
train_labels = []

def load_all():
    img, lb = load_training()
    train_X = np.array(img);
    train_y = np.array(lb);

    img, lb = load_testing()
    test_X = np.array(img);
    test_y = np.array(lb);
    return train_X, train_y, test_X, test_y


def load_testing():
    ims, labels = load(os.path.join(path, test_img_fname),
                     os.path.join(path, test_lbl_fname))

    test_images = ims
    test_labels = labels

    return ims, labels


def load_training():
    ims, labels = load(os.path.join(path, train_img_fname),
                     os.path.join(path, train_lbl_fname))

    train_images = ims
    train_labels = labels

    return ims, labels


def load(path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                'got %d' % magic)

        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                'got %d' % magic)

        image_data = array("B", file.read())

    images = []
    for i in xrange(size):
        images.append([0]*rows*cols)

    for i in xrange(size):
        images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

    return images, labels


if __name__ == "__main__":
    print 'Testing'
    itrain, lbtrain = load_training()
    itest, lbtest = load_testing()
    print 'Trainning data:', len(itrain), '*', len(itrain[0]), len(lbtrain)
    print 'Testing data:', len(itest), '*', len(itest[0]), len(lbtest)