import numpy as np
class softmax():
    def __init__(self):
        pass

    def forward(self,input):
        self.input =input
        y=np.max(self.input, axis=1,keepdims=True)
        exp=np.exp(self.input-y)
        return exp/np.sum(exp,axis=1,keepdims=True)
    def backward(self, gradient):
        return gradient #mostly used alongside cross entropy loss casuing gradient simplification to output-true_pred
    

class RELU():

    def __init__(self):
        pass

    def forward(self,input):
        self.input =input
        return np.maximum(0, self.input)
    def backward(self, gradient):
        return gradient*(self.input>0)

class Leaky_RELU():
    def __init__(self,lr=0.01):
        self.lr=lr
    def forward(self,input):
        return np.where(self.input>0,self.input, self.lr*self.input)
    def backward(self, gradient):
        return np.where(self.input>0,gradient,self.lr*gradient)
    
