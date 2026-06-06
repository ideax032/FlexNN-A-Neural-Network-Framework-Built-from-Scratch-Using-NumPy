import numpy as np 
from .mode import model_mode 
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
        if model_mode.mode:
            self.mask=np.random.rand(*x.shape)>self.prob
            return x*self.mask
        else:
            return x
    def backward(self,gradient):
        output=gradient*self.mask
        self.mask=None
        return output
    


        