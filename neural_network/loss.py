import numpy as np
class cross_entropy():
    def forward(self,output,y,epsilon=1e-15):
        m=y.shape[0]
        output=np.clip(output,epsilon,1-epsilon)
        loss=-np.sum(y*np.log(output+epsilon))/m
        return loss
    def backward(self,output,y):
        grad=output-y #as used alone with softmax so the gradient is output-y so we just tranfer the gradient
        return grad