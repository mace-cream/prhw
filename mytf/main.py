import numpy as np
import os
from mytf.utils import _sigmoid,_softmax

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
        elif self.op_type=='relu':
            result = self.input_list[0].eval(feed)
            result[result<0] = 0
        elif self.op_type=='softmax':
            result = _softmax(self.input_list[0].eval(feed))
        elif self.op_type=='log_softmax':
            logit = self.input_list[0].eval(feed)
            # Note: The exact calculation of log(sum(exp(s_i))) has serious numerical issue, we use max instead.
            result = logit - np.log(np.sum(np.exp(logit),1,keepdims=True))
            if np.any(~np.isfinite(result)):
                result = logit - np.max(logit,1,keepdims=True)
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
        elif self.op_type=='imagePadding':
            pad_size = int(self.input_list[1])
            result = np.pad(self.input_list[0].eval(feed), ((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)), 'constant')
        elif self.op_type=='imageZIP':
            kernel_size,stride = self.input_list[1],self.input_list[2]
            pad = int((kernel_size-1)/2)
            value = self.input_list[0].eval(feed)
            zipped = []
            for i in range(pad, self.input_list[0].shape[1]-pad, stride):
                for j in range(pad, self.input_list[0].shape[2]-pad, stride):
                    zipped.append(value[:,i-pad:i+pad+1,j-pad:j+pad+1,:].reshape((self.input_list[0].shape[0],-1)))
            result = np.stack(zipped,0).transpose((1,0,2)).reshape(self.shape)
        elif self.op_type=='reshape':
            result = self.input_list[0].eval(feed).reshape(self.input_list[1])

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
            elif out.op_type=='relu':
                forward_gradient = out.back(target,feed)
                local_gradient = out.eval(feed)
                local_gradient = (local_gradient!=0)*1.0
                gradient = gradient + local_gradient
            elif out.op_type=='softmax':
                logits = _softmax(self.eval(feed))
                forward_gradient = out.back(target,feed)
                if forward_gradient!=0:
                    local_gradient = []
                    for i in range(self.shape[0]):
                        local_logits = logits[i].reshape((-1,1))
                        jacob = np.diag(logits[i])-np.matmul(local_logits,local_logits.T)
                        local_gradient.append(np.matmul(forward_gradient[i].reshape((1,-1)),jacob))
                    local_gradient = np.concatenate(local_gradient,0)
                    gradient = gradient + local_gradient
            elif out.op_type=='log_softmax':
                logits = _softmax(self.eval(feed))
                forward_gradient = out.back(target,feed)
                local_gradient = []
                for i in range(self.shape[0]):
                    local_logits = logits[i].reshape((-1,1))
                    jacob = np.eye(local_logits.shape[0])-np.matmul(local_logits,(0*local_logits+1).T)
                    local_gradient.append(np.matmul(jacob,forward_gradient[i].reshape((-1,1))).T)
                local_gradient = np.concatenate(local_gradient,0)
                gradient = gradient + local_gradient
            elif out.op_type=='add':
                dim_difference = len(out.shape)-len(self.shape)
                boardcast_dim = [i for i in range(dim_difference)] + [i for i in range(dim_difference, len(out.shape)) if self.shape[i-dim_difference]==1]
                if self is out.input_list[0]:
                    gradient = gradient + np.sum(out.back(target,feed),tuple(boardcast_dim)).reshape(self.shape)
                if self is out.input_list[1]:
                    gradient = gradient + np.sum(out.back(target,feed),tuple(boardcast_dim)).reshape(self.shape)
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
            elif out.op_type=='imagePadding':
                gradient = gradient + out.back(target,feed)[:,out.input_list[1]:-out.input_list[1],out.input_list[1]:-out.input_list[1],:]
            elif out.op_type=='imageZIP':
                forward_gradient = out.back(target,feed).reshape((self.shape[0],-1,out.shape[-1])).transpose((1,0,2))
                sub_gradient = np.zeros(self.shape)
                kernel_size,stride = out.input_list[1],out.input_list[2]
                pad = int((kernel_size-1)/2)
                counter = 0
                for i in range(pad, self.shape[1]-pad, stride):
                    for j in range(pad, self.shape[2]-pad, stride):
                        sub_gradient[:,i-pad:i+pad+1,j-pad:j+pad+1,:] += forward_gradient[counter].reshape((self.shape[0],kernel_size,kernel_size,-1))
                        counter = counter + 1
                gradient = gradient + sub_gradient
            elif out.op_type=='reshape':
                gradient = gradient + out.back(target,feed).reshape(self.shape)

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

class Graph(object):
    def __init__(self,config):
        self.construct_model(config)
        self.initWeight()

    def construct_model(self,config):
        self.weight = {}
        self.weight_value = {}

    def initWeight(self,initializer=np.random.standard_normal):
        self.weight_value = {k:initializer(v.shape) for k,v in self.weight.items()}

    def saveModel(self,path):
        filepath, _ = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.savez(path, **(self.weight_value))

    def loadModel(self,path):
        if path[-4:] != '.npz':
            path = path + '.npz'
        self.weight_value = dict(np.load(path))
