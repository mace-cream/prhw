from utils import load_mnist
import numpy as np
import matplotlib.pyplot as plt

# Set target classes
CLASS_POSTIVE = 0
CLASS_NEGATIVE = 1

# Load Data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train/255.0
x_test = x_test/255.0

def Perceptron(x_train, y_train, x_test, y_test, CLASS_POSTIVE, CLASS_NEGATIVE):
    # Process Data
    x_train = np.concatenate([x_train[y_train==CLASS_POSTIVE],x_train[y_train==CLASS_NEGATIVE]],0)
    x_train = np.concatenate([x_train,np.ones((x_train.shape[0],1))],1)
    y_train = np.concatenate([np.ones((np.sum(y_train==CLASS_POSTIVE),)), -np.ones((np.sum(y_train==CLASS_NEGATIVE),))], 0)
    x_train = (x_train.T*y_train).T
    x_test = np.concatenate([x_test[y_test==CLASS_POSTIVE],x_test[y_test==CLASS_NEGATIVE]],0)
    x_test = np.concatenate([x_test,np.ones((x_test.shape[0],1))],1)
    y_test = np.concatenate([np.ones((np.sum(y_test==CLASS_POSTIVE),)), -np.ones((np.sum(y_test==CLASS_NEGATIVE),))], 0)
    x_test = (x_test.T*y_test).T

    # Training
    w = np.zeros((x_train.shape[1],))
    epoch_num = 100
    learning_rate = 0.1
    for e in range(epoch_num):
        target = np.sum(x_train*w,1)<=0
        if np.mean(target==0):
            break
        w = w + learning_rate*np.mean(x_train[target],0)

    #Testing
    accuracy = np.mean(np.sum(x_test*w,1)>0)

    return accuracy

if __name__=="__main__":
    pass

_debug = np.array([2,3,3])

