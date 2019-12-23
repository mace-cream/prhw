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
        if self.name in feed.keys():
            return feed[self.name]
        if self.op_type is None:
            result = feed[self.name]
        elif self.op_type is 'Constant':
            result = self.value
        elif self.op_type=='matmul':
            result = np.matmul(self.input_list[0].eval(feed),self.input_list[1].eval(feed))
        elif self.op_type=='sigmoid':
            result = _sigmoid(self.input_list[0].eval(feed))
        elif self.op_type=='softmax':
            result = _softmax(self.input_list[0].eval(feed))
        elif self.op_type=='add':
            result = self.input_list[0].eval(feed)+self.input_list[1].eval(feed)
        elif self.op_type=='log':
            result = np.log(self.input_list[0].eval(feed))
        elif self.op_type=='product':
            result = self.input_list[0].eval(feed)*self.input_list[1].eval(feed)
        elif self.op_type=='reduce_sum':
            result = np.sum(self.input_list[0].eval(feed))
        elif self.op_type=='scale':
            result = self.input_list[1]*self.input_list[0].eval(feed)
        elif self.op_type=='accuracy':
            result = np.mean(np.argmax(self.input_list[0].eval(feed),-1)==np.argmax(self.input_list[1].eval(feed),-1))
        
        feed.update({self.name: result})
        return result

    def back(self,target,feed):
        '''
        Define the gradient back propgation with respect to 'target' given input 'feed'
        '''
        if self.name+'_g' in feed.keys():
            return feed[self.name+'_g']
        if self is target:
            return np.ones(self.shape)
        gradient = 0
        for out in self.output_list:
            if out.op_type=='matmul':
                if self is out.input_list[0]:
                    gradient = gradient + np.matmul(out.back(target,feed), out.input_list[1].eval(feed).T)
                if self is out.input_list[1]:
                    gradient = gradient + np.matmul(out.input_list[0].eval(feed).T, out.back(target,feed))
            elif out.op_type=='sigmoid':
                jacob = _sigmoid(self.eval(feed)) * (1-_sigmoid(self.eval(feed)))
                gradient = gradient + jacob * out.back(target,feed)
            elif out.op_type=='softmax':
                logits = _softmax(self.eval(feed))
                forward_gradient = out.back(target,feed)
                local_gradient = []
                for i in range(self.shape[0]):
                    local_logits = logits[i].reshape((-1,1))
                    jacob = np.diag(logits[i])-np.matmul(local_logits,local_logits.T)
                    local_gradient.append(np.matmul(forward_gradient[i].reshape((1,-1)),jacob))
                local_gradient = np.concatenate(local_gradient,0)
                gradient = gradient + local_gradient
            elif out.op_type=='add':
                if self is out.input_list[0]:
                    gradient = gradient + out.back(target,feed)
                if self is out.input_list[1]:
                    gradient = gradient + out.back(target,feed)
            elif out.op_type=='log':
                gradient = gradient + 1/self.eval(feed)*out.back(target,feed)
            elif out.op_type=='product':
                if self is out.input_list[0]:
                    gradient = gradient + out.back(target,feed)*out.input_list[1].eval(feed)
                if self is out.input_list[1]:
                    gradient = gradient + out.back(target,feed)*out.input_list[0].eval(feed)
            elif out.op_type=='reduce_sum':
                gradient = gradient + np.ones(self.shape)*out.back(target,feed)
            elif out.op_type=='scale':
                gradient = gradient + out.input_list[1]*out.back(target,feed)
            elif out.op_type in ['accuracy']:
                pass
        
        feed.update({self.name+'_g': gradient})
        return gradient

class NameManager(object):
    def __init__(self):
        self.nameList = {}
    def get(self,name):
        if name not in self.nameList.keys():
            self.nameList.update({name: 0})
        else:
            self.nameList.update({name: self.nameList[name]+1})
        return name+'_'+str(self.nameList[name])

NM = NameManager()

def matmul(x1,x2):
    out = tensor(x1.shape[:-1]+x2.shape[1:],NM.get('matmul'),'matmul',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def add(x1,x2):
    out = tensor(x1.shape,NM.get('add'),'add',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,NM.get('sigmoid'),'sigmoid',[x])
    x.output_list.append(out)
    return out

def log(x):
    out = tensor(x.shape,NM.get('log'),'log',[x])
    x.output_list.append(out)
    return out

def product(x1,x2):
    out = tensor(x1.shape,NM.get('product'),'product',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def softmax(x):
    out = tensor(x.shape,NM.get('softmax'),'softmax',[x])
    x.output_list.append(out)
    return out

def reduce_sum(x):
    out = tensor([1,1],NM.get('reduce_sum'),'reduce_sum',[x])
    x.output_list.append(out)
    return out

def scale(x,alpha):
    out = tensor(x.shape,NM.get('scale'),'scale',[x,alpha])
    x.output_list.append(out)
    return out

def reduce_mean(x):
    out = scale(reduce_sum(x),1/np.product(x.shape))
    return out

def CE(x,y):
    out = scale(reduce_sum(product(y,log(x))),-1)
    return out

def accuracy(pred,y):
    out = tensor([1,1],NM.get('accuracy'),'accuracy',[pred,y])
    pred.output_list.append(out)
    y.output_list.append(out)
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
    c = add(matmul(e,d),scale(CE(a2,a2),-1))

    feed = {'a':np.array([[1.,2],[3,4.5]]),'b':np.array([[1.],[2]]),'c':np.array([[1.,2]])}
    print(a.back(c,feed))
    import copy
    for i in range(2):
        for j in range(2):
            feed2 = copy.deepcopy(feed)
            feed2['a'][i][j] = feed2['a'][i][j]+2e-3
            print((c.eval(feed2)-c.eval(feed))/2e-3)