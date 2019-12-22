import numpy as np
import mytf
from utils import load_mnist

# Load Data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train/255.0
x_test = x_test/255.0

# Define the network

_debug = np.array([2,3,3])