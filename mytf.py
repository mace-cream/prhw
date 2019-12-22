import numpy as np
from op import _sigmoid

class tensor(object):
    def __init__(self,shape,name,op_type=None,input_list=None):
        self.shape = shape
        self.name = name
        self.op_type = op_type
        self.input_list = input_list
        self.output_list = []
    
    def eval(self,feed):
        if self.op_type is None:
            return feed[self.name]
        if self.op_type=='matmul':
            return np.matmul(self.input_list[0].eval(feed),self.input_list[1].eval(feed))
        if self.op_type=='sigmoid':
            return _sigmoid(self.input_list[0].eval(feed))

    def back(self,target,feed):
        if self is target:
            return np.ones(self.shape)
        gradient = 0
        for out in self.output_list:
            if out.op_type=='matmul':
                if self is out.input_list[0]:
                    gradient = gradient + np.matmul(out.back(target,feed), out.input_list[1].eval(feed).T)
                if self is out.input_list[1]:
                    gradient = gradient + np.matmul(out.input_list[0].eval(feed).T, out.back(target,feed))
            if out.op_type=='sigmoid':
                gradient = gradient + _sigmoid(out.input_list[0].eval(feed)) * (1-_sigmoid(out.input_list[0].eval(feed))) * out.back(target,feed)
        return gradient

def matmul(x1,x2):
    out = tensor(x1.shape[:-1]+x2.shape[1:],'name','matmul',[x1,x2])
    x1.output_list.append(out)
    x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,'name','sigmoid',[x])
    x.output_list.append(out)
    return out

def softmax(x):
    out = tensor(x.shape,'name','softmax',[x])
    x.output_list.append(out)
    return out