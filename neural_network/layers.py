import numpy as np
class Dense():
    def __init__(self, input_size,output_size):
        self.w=np.random.randn(input_size,output_size)*(np.sqrt(2/input_size))
        self.b=np.zeros((1,output_size), dtype=np.float32)

    def forward(self,x):
        self.input=x
        return np.dot(x,self.w)+self.b

    def backward(self,gradient):
        m=self.input.shape[0]
        self.dw=np.dot(self.input.T,gradient)/m
        self.db=np.sum(gradient,axis=0,keepdims=True)/m
        self.optimizer.update(self)

        return np.dot(gradient,self.w.T)     

    
    def show(self):
        print(self.w.shape)
        print(self.b.shape)    