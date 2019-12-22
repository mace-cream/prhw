import numpy as np

def _sigmoid(x):
    return 1/(1+np.exp(-x))