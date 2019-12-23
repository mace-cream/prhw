import numpy as np
import mytf
from utils import load_mnist, one_hot
from model import MLP

# Load Data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train/255.0
x_test = x_test/255.0

# Define the network
config = {'EpochNum': 100, 'BatchSize': 50, 'InputDim': 784, 'OutputDim': 10, 'LayerNum': 1, 'HiddenNum': [100]}
model = MLP(config)

# Training


_debug = np.array([2,3,3])