import numpy as np
import mytf
from utils import load_mnist, one_hot
from model import MLP

# Load Data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train/255.0
x_test = x_test/255.0

# Define the network
config = {'EpochNum': 100, 'BatchSize': 50, 'InputDim': 784, 'OutputDim': 10, 'LayerNum': 1, 'HiddenNum': [100], 'LearningRate': 0.001}
model = MLP(config)

# Training
for e in range(config['EpochNum']):
    perm = np.linspace(0,x_train.shape[0]-1,x_train.shape[0],dtype=int)
    np.random.shuffle(perm)
    x_train = x_train[perm]
    y_train = y_train[perm]
    counter = 0
    loss = []
    while counter + config['BatchSize'] <= x_train.shape[0]:
        x_batch = x_train[counter:counter+config['BatchSize']].reshape([config['BatchSize'],-1])
        y_batch = one_hot(y_train[counter:counter+config['BatchSize']], config['OutputDim'])
        feed = {model.input.name: x_batch, model.label.name: y_batch}
        feed.update(model.weight_value)
        loss.append(model.loss.eval(feed))
        gradient = {k:v.back(model.loss,feed) for k,v in model.weight.items()}
        model.weight_value.update({
            k:model.weight_value[k]-config['LearningRate']*gradient[k] for k in model.weight.keys()})
        counter = counter + config['BatchSize']
    
    counter = 0
    accuracy = []
    while counter + config['BatchSize'] <= x_test.shape[0]:
        x_batch = x_train[counter:counter+config['BatchSize']].reshape([config['BatchSize'],-1])
        y_batch = one_hot(y_train[counter:counter+config['BatchSize']], config['OutputDim'])
        feed = {model.input.name: x_batch, model.label.name: y_batch}
        feed.update(model.weight_value)
        accuracy.append(model.accuracy.eval(feed))
        counter = counter + config['BatchSize']

    print('Epoch ',e,' Done, Loss: ',round(np.mean(loss),4),', Test Accuracy: ',round(np.mean(accuracy),4))




_debug = np.array([2,3,3])