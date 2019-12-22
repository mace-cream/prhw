import numpy as np

def _sigmoid(x):
    return 1/(1+np.exp(-x))

def _softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x),1)).T

if __name__=="__main__":
    a = np.arange(4).reshape((2,2))
    print(_softmax(a))