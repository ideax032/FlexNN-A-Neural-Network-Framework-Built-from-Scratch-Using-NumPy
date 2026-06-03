import numpy as np 

class batch_normalization():
    def __init__(self):
        self.w=None
        self.b=None
    
    def __name__(self):
        return "Batch_Normalization"
    
    def forward(self, x):
        pass

class Dropout():
    def __init__(self,prob=0.2):
        self.prob=prob
        self.mask=None
    
    def __name__(self):
        return "Dropout"
    
    def forward(self,x):
        self.mask=np.random.rand(*x.shape)>self.prob
        return x*self.mask
    def backward(self,gradient):
        return gradient*self.mask

        