import numpy as np
from op import _sigmoid,_softmax

class tensor(object):
    def __init__(self,shape,name,op_type=None,input_list=None,value=None):
        self.shape = shape
        self.name = name
        self.op_type = op_type
        self.input_list = input_list
        self.output_list = []
        self.value = value
    
    def eval(self,feed):
        '''
        Define the forward computation given input 'feed'
        '''
        if self.op_type is None:
            return feed[self.name]
        if self.op_type is 'Constant':
            return self.value
        if self.op_type=='matmul':
            return np.matmul(self.input_list[0].eval(feed),self.input_list[1].eval(feed))
        if self.op_type=='sigmoid':
            return _sigmoid(self.input_list[0].eval(feed))
        if self.op_type=='softmax':
            return _softmax(self.input_list[0].eval(feed))
        if self.op_type=='add':
            return self.input_list[0].eval(feed)+self.input_list[1].eval(feed)
        if self.op_type=='log':
            return np.log(self.input_list[0].eval(feed))
        if self.op_type=='product':
            return self.input_list[0].eval(feed)*self.input_list[1].eval(feed)
        if self.op_type=='reduce_sum':
            return np.sum(self.input_list[0].eval(feed))

    def back(self,target,feed):
        '''
        Define the gradient back propgation with respect to 'target' given input 'feed'
        '''
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
                jacob = _sigmoid(self.eval(feed)) * (1-_sigmoid(self.eval(feed)))
                gradient = gradient + jacob * out.back(target,feed)
            if out.op_type=='softmax':
                logits = _softmax(self.eval(feed))
                forward_gradient = out.back(target,feed)
                local_gradient = []
                for i in range(self.shape[0]):
                    local_logits = logits[i].reshape((-1,1))
                    jacob = np.diag(logits[i])-np.matmul(local_logits,local_logits.T)
                    local_gradient.append(np.matmul(forward_gradient[i].reshape((1,-1)),jacob))
                local_gradient = np.concatenate(local_gradient,0)
                gradient = gradient + local_gradient
            if out.op_type=='add':
                if self is out.input_list[0]:
                    gradient = gradient + out.back(target,feed)
                if self is out.input_list[1]:
                    gradient = gradient + out.back(target,feed)
            if out.op_type=='log':
                gradient = gradient + 1/self.eval(feed)*out.back(target,feed)
            if out.op_type=='product':
                if self is out.input_list[0]:
                    gradient = gradient + out.back(target,feed)*out.input_list[1].eval(feed)
                if self is out.input_list[1]:
                    gradient = gradient + out.back(target,feed)*out.input_list[0].eval(feed)
            if out.op_type=='reduce_sum':
                gradient = gradient + np.ones(self.shape)*out.back(target,feed)
        return gradient

def matmul(x1,x2):
    out = tensor(x1.shape[:-1]+x2.shape[1:],'name','matmul',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def add(x1,x2):
    out = tensor(x1.shape,'name','add',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,'name','sigmoid',[x])
    x.output_list.append(out)
    return out

def log(x):
    out = tensor(x.shape,'name','log',[x])
    x.output_list.append(out)
    return out

def product(x1,x2):
    out = tensor(x1.shape,'name','product',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def softmax(x):
    out = tensor(x.shape,'name','softmax',[x])
    x.output_list.append(out)
    return out

def reduce_sum(x):
    out = tensor([1,1],'name','reduce_sum',[x])
    x.output_list.append(out)
    return out

def CE(x,y):
    out = reduce_sum(product(y,log(x)))
    return out


if __name__=="__main__":
    '''
    Here we use a very simple example to check the result, with the gradient result given by difference limit.
    '''
    a = tensor([2,2],'a')
    b = tensor([2,1],'b')
    c = tensor([1,2],'c')
    a2 = sigmoid(product(a,a))
    d = sigmoid(matmul(a2,b))
    d = add(d,matmul(a2,b))
    e = sigmoid(matmul(c,a2))
    c = add(matmul(e,d),CE(a2,a2))

    feed = {'a':np.array([[1.,2],[3,4.5]]),'b':np.array([[1.],[2]]),'c':np.array([[1.,2]])}
    print(a.back(c,feed))
    import copy
    for i in range(2):
        for j in range(2):
            feed2 = copy.deepcopy(feed)
            feed2['a'][i][j] = feed2['a'][i][j]+2e-3
            print((c.eval(feed2)-c.eval(feed))/2e-3)