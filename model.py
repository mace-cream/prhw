import numpy as np
import mytf

class MLP(object):
    def __init__(self,config):
        self.construct_model(config)
        self.initWeight()

    def construct_model(self,config):
        self.weight = {}
        self.weight_value = {}
        self.input = mytf.tensor([config['BatchSize'], config['InputDim']],'Input')
        self.label = mytf.tensor([config['BatchSize'], config['OutputDim']],'Label')
        h = self.input
        for i in range(config['LayerNum']):
            if i==0:
                w = mytf.tensor([config['InputDim'], config['HiddenNum'][i]],'Weight'+str(i))
                self.weight['Weight'+str(i)] = w
            else:
                w = mytf.tensor([config['HiddenNum'][i-1], config['HiddenNum'][i]],'Weight'+str(i))
                self.weight['Weight'+str(i)] = w
            b = mytf.tensor([1, config['HiddenNum'][i]],'Bias'+str(i))
            self.weight['Bias'+str(i)] = b
            h = mytf.add(mytf.matmul(h, w), b)
            h = mytf.sigmoid(h)
        w = mytf.tensor([config['HiddenNum'][-1], config['OutputDim']],'OutputWeight')
        self.weight['OutputWeight'] = w
        b = mytf.tensor([1, config['OutputDim']],'OutputBias')
        self.weight['OutputBias'] = b
        h = mytf.add(mytf.matmul(h, w), b)
        self.out = mytf.softmax(h)
        self.loss = mytf.CE_with_logit(h, self.label)
        for w in self.weight.values():
            self.loss = mytf.add(self.loss, mytf.scale(mytf.reduce_mean(mytf.product(w,w)),config['lambda']))
        self.accuracy = mytf.accuracy(self.out, self.label)
    
    def initWeight(self,initializer=np.random.standard_normal):
        self.weight_value = {k:initializer(v.shape) for k,v in self.weight.items()}

class LeNet(object):
    def __init__(self,config):
        self.construct_model(config)
        self.initWeight()

    def construct_model(self,config):
        self.weight = {}
        self.weight_value = {}
        self.input = mytf.tensor([config['BatchSize'], config['Height'], config['Width'], config['InputDim']],'Input')
        self.label = mytf.tensor([config['BatchSize'], config['OutputDim']],'Label')
        h = self.input
        for i in range(config['LayerNum']):
            if i==0:
                w = mytf.tensor([config['KernelSize'], config['KernelSize'], config['InputDim'], config['HiddenNum'][i]],'Weight'+str(i))
                self.weight['Weight'+str(i)] = w
            else:
                w = mytf.tensor([config['KernelSize'], config['KernelSize'], config['HiddenNum'][i-1], config['HiddenNum'][i]],'Weight'+str(i))
                self.weight['Weight'+str(i)] = w
            b = mytf.tensor([1, config['HiddenNum'][i]],'Bias'+str(i))
            self.weight['Bias'+str(i)] = b
            h = mytf.add(mytf.conv2D(h, w), b)
            h = mytf.sigmoid(h)
        self.afterconv = h
        h = mytf.reshape(h,[config['BatchSize'], int(np.prod(h.shape)/config['BatchSize'])])
        w = mytf.tensor([h.shape[-1], config['OutputDim']],'OutputWeight')
        self.weight['OutputWeight'] = w
        b = mytf.tensor([1, config['OutputDim']],'OutputBias')
        self.weight['OutputBias'] = b
        h = mytf.add(mytf.matmul(h, w), b)
        self.aftermap = h
        self.out = mytf.softmax(h)
        self.loss = mytf.CE_with_logit(h, self.label)
        for w in self.weight.values():
            self.loss = mytf.add(self.loss, mytf.scale(mytf.reduce_mean(mytf.product(w,w)),config['lambda']))
        self.accuracy = mytf.accuracy(self.out, self.label)

    def initWeight(self,initializer=np.random.standard_normal):
        self.weight_value = {k:initializer(v.shape) for k,v in self.weight.items()}