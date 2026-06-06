import numpy as np
from .mode import model_mode

class Dense():
    def __init__(self, input_size,output_size):
        self.w=np.random.randn(input_size,output_size)*(np.sqrt(2/input_size))
        self.b=np.zeros((1,output_size), dtype=np.float32)
    
    def __name__(self):
        return "Dense"
    
    def forward(self,x):
        if model_mode.mode:
            self.input=x
        return np.dot(x,self.w)+self.b

    def backward(self,gradient):
        m=self.input.shape[0]
        self.dw=np.dot(self.input.T,gradient)/m
        self.db=np.sum(gradient,axis=0,keepdims=True)/m
        self.optimizer.update(self)
        output=np.dot(gradient,self.w.T)
        self.input=None
        self.dw=None
        self.db=None
        
        return output

    
    def show(self):
        print(self.w.shape)
        print(self.b.shape)    