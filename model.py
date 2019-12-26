import numpy as np
import mytf

class MLP(mytf.Graph):
    def __init__(self,config):
        super().__init__(config)

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
        if config['lambda']!=0:
            for w in self.weight.values():
                self.loss = mytf.add(self.loss, mytf.scale(mytf.reduce_mean(mytf.product(w,w)),config['lambda']))
        self.accuracy = mytf.accuracy(self.out, self.label)

class LeNet(mytf.Graph):
    def __init__(self,config):
        super().__init__(config)

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
        h = mytf.reshape(h,[config['BatchSize'], int(np.prod(h.shape)/config['BatchSize'])])
        w = mytf.tensor([h.shape[-1], config['HiddenNumfc']],'OutputWeightfc')
        self.weight['OutputWeightfc'] = w
        b = mytf.tensor([1, config['HiddenNumfc']],'OutputBiasfc')
        self.weight['OutputBiasfc'] = b
        h = mytf.add(mytf.matmul(h, w), b)
        h = mytf.sigmoid(h)
        w = mytf.tensor([h.shape[-1], config['OutputDim']],'OutputWeight')
        self.weight['OutputWeight'] = w
        b = mytf.tensor([1, config['OutputDim']],'OutputBias')
        self.weight['OutputBias'] = b
        h = mytf.add(mytf.matmul(h, w), b)
        self.out = mytf.softmax(h)
        self.loss = mytf.CE_with_logit(h, self.label)
        if config['lambda']!=0:
            for w in self.weight.values():
                self.loss = mytf.add(self.loss, mytf.scale(mytf.reduce_mean(mytf.product(w,w)),config['lambda']))
        self.accuracy = mytf.accuracy(self.out, self.label)