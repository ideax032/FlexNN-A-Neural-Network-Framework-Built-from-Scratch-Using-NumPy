import numpy as np
from .mode import model_mode 
class softmax():
    def __init__(self):
        pass
    
    def __name__(self):
        return "Softmax"
    
    def forward(self,input):
        if model_mode.mode:
            self.input =input
        y=np.max(input, axis=1,keepdims=True)
        exp=np.exp(input-y)
        return exp/np.sum(exp,axis=1,keepdims=True)
    
    def backward(self, gradient):
        return gradient #mostly used alongside cross entropy loss casuing gradient simplification to output-true_pred
    
class RELU():

    def __init__(self):
        pass
    
    def __name__(self):
        return "RELU"

    def forward(self,input):
        if model_mode.mode:
            self.input =input
        return np.maximum(0, input)
    
    def backward(self, gradient):
        output=gradient*(self.input>0)
        self.input=None
        return output

class Leaky_RELU():
    def __init__(self,lr=0.01):
        self.lr=lr
    
    def __name__(self):
        return "Leaky_RELU"
    
    def forward(self,input):
        if model_mode.mode:
            self.input =input
        return np.where(input>0,input, self.lr*input)
    
    def backward(self, gradient):
        output=np.where(self.input>0,gradient,self.lr*gradient)
        self.input=None
        return output

class Sigmoid():
    def __init__(self):
        pass 
    
    def __name__(self):
        return "Sigmoid"

    def forward(self,x):
        x=np.clip(x, -500, 500)
        output=1/(np.exp(-x)+1)
        if model_mode.mode:
            self.output=output
        return output
    
    def backward(self,gradient):
        output=self.output
        self.output=None

        return output  #could lead to vanishing graident problem for deep neural network 


class Tanh():
    def __init__(self):
        pass
    
    def  __name__(self):
        return "Tanh"
    
    def forward(self,x):
        x=np.clip(x,-500,500)
        exp_x=np.exp(x)
        exp_neg_x=np.exp(-x)
        output=(exp_x-exp_neg_x)/(exp_x+exp_neg_x) #similar to sigmoid function but ranges from -1 to 1
        if model_mode.mode:
            self.output=output 
        return output

    def backward(self,gradient):
        output=gradient*(1-self.output**2) #derivative of tanh is 1-output^2
        self.output=None
        return output 
        
    