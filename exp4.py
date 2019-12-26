import numpy as np
import time
from utils import load_mnist
from mytf import one_hot
from model import MLP, LeNet

task_name = str(int(time.time()))

# Load Data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train/255.0
x_test = x_test/255.0

# Define the network
Network = ['MLP', 'LeNet'][1]
if Network=='MLP':
    config = {'EpochNum': 100, 'BatchSize': 50, 'InputDim': 784, 'OutputDim': 10, 'LayerNum': 3, 'HiddenNum': [1000,500,100], 'LearningRate': 0.01, 'lambda': 0.1}
    model = MLP(config)
if Network=='LeNet':
    config = {'EpochNum': 100, 'BatchSize': 100, 'Height': 28, 'Width': 28, 'InputDim': 1, 'OutputDim': 10, 'LayerNum': 2, 'HiddenNum': [32,64,128], 'HiddenNumfc': 1000, 'KernelSize': 3, 'LearningRate': 0.3, 'lambda': 0.1}
    model = LeNet(config)

# Training
for e in range(config['EpochNum']):
    start_time = time.time()
    perm = np.linspace(0,x_train.shape[0]-1,x_train.shape[0],dtype=int)
    np.random.shuffle(perm)
    x_train = x_train[perm]
    y_train = y_train[perm]
    counter = 0
    loss = []
    accuracy_train = []
    while counter + config['BatchSize'] <= x_train.shape[0]:
        if Network=='MLP':
            x_batch = x_train[counter:counter+config['BatchSize']].reshape([config['BatchSize'],-1])
            y_batch = one_hot(y_train[counter:counter+config['BatchSize']], config['OutputDim'])
            feed = {model.input.name: x_batch, model.label.name: y_batch}
        if Network=='LeNet':
            x_batch = x_train[counter:counter+config['BatchSize']].reshape([config['BatchSize'], config['Height'], config['Width'], config['InputDim']])
            y_batch = one_hot(y_train[counter:counter+config['BatchSize']], config['OutputDim'])
            feed = {model.input.name: x_batch, model.label.name: y_batch}
        feed.update(model.weight_value)
        loss.append(model.loss.eval(feed))
        accuracy_train.append(model.accuracy.eval(feed))
        model.update(feed,config['LearningRate'])
        counter = counter + config['BatchSize']
    
    counter = 0
    accuracy_test = []
    while counter + config['BatchSize'] <= x_test.shape[0]:
        if Network=='MLP':
            x_batch = x_test[counter:counter+config['BatchSize']].reshape([config['BatchSize'],-1])
            y_batch = one_hot(y_test[counter:counter+config['BatchSize']], config['OutputDim'])
            feed = {model.input.name: x_batch, model.label.name: y_batch}
        if Network=='LeNet':
            x_batch = x_test[counter:counter+config['BatchSize']].reshape([config['BatchSize'], config['Height'], config['Width'], config['InputDim']])
            y_batch = one_hot(y_test[counter:counter+config['BatchSize']], config['OutputDim'])
            feed = {model.input.name: x_batch, model.label.name: y_batch}
        feed.update(model.weight_value)
        accuracy_test.append(model.accuracy.eval(feed))
        counter = counter + config['BatchSize']

    process_speed = round(counter/(time.time()-start_time),1)
    print('Epoch ',e,' Done, Loss: ',round(np.mean(loss),4),
        ', Train Acc: ',round(np.mean(accuracy_train),4),
        ', Test Acc: ',round(np.mean(accuracy_test),4),
        ', FPS: ',process_speed)
    model.saveModel('Model/'+task_name+'/'+str(e))

_debug = np.array([2,3,3])