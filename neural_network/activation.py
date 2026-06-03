import numpy as np
class softmax():
    def __init__(self):
        pass
    
    def __name__(self):
        return "Softmax"
    
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
    
    def __name__(self):
        return "RELU"

    def forward(self,input):
        self.input =input
        return np.maximum(0, self.input)
    
    def backward(self, gradient):
        return gradient*(self.input>0)

class Leaky_RELU():
    def __init__(self,lr=0.01):
        self.lr=lr
    
    def __name__(self):
        return "Leaky_RELU"
    
    def forward(self,input):
        return np.where(self.input>0,self.input, self.lr*self.input)
    
    def backward(self, gradient):
        return np.where(self.input>0,gradient,self.lr*gradient)
    
class Sigmoid():
    def __init__(self):
        pass 
    
    def __name__(self):
        return "Sigmoid"

    def forward(self,x):
        x=np.clip(x, -500, 500)
        self.output=1/(np.exp(-x)+1)
        return self.output
    
    def backward(self,gradient):
        return gradient*self.output*(1-self.output)  #could lead to vanishing graident problem for deep neural network 


class Tanh():
    def __init__(self):
        pass
    
    def  __name__(self):
        return "Tanh"
    
    def forward(self,x):
        x=np.clip(x,-500,500)
        exp_x=np.exp(x)
        exp_neg_x=np.exp(-x)
        self.output=(exp_x-exp_neg_x)/(exp_x+exp_neg_x) #similar to sigmoid function but ranges from -1 to 1 
        return self.output

    def backward(self,gradient):
        return gradient*(1-self.output**2)
        
    