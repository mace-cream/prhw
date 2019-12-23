import numpy as np
import mytf

class MLP(object):
    def __init__(self,config):
        self.input = mytf.tensor([config['BatchSize'], config['InputDim']],'Input')
        self.label = mytf.tensor([config['BatchSize'], config['OutputDim']],'Label')
        h = self.input
        for i in range(config['LayerNum']):
            if i==0:
                w = mytf.tensor([config['InputDim'], config['HiddenNum'][i]],'Weight'+str(i))
            else:
                w = mytf.tensor([config['HiddenNum'][i-1], config['HiddenNum'][i]],'Weight'+str(i))
            b = mytf.tensor([1, config['HiddenNum'][i]],'Bias'+str(i))
            h = mytf.add(mytf.matmul(h, w), b)
            h = mytf.sigmoid(h)
        w = mytf.tensor([config['HiddenNum'][-1], config['OutputDim']],'OutputWeight')
        b = mytf.tensor([1, config['OutputDim']],'OutputBias')
        h = mytf.add(mytf.matmul(h, w), b)
        self.out = mytf.softmax(h)
        self.loss = mytf.CE(self.out, self.label)