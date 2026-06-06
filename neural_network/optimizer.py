import numpy as np
class Adam():
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999, epsilon=1e-8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.t=0
    
    def __name__(self):
        return "Adam"
    
    def pre_update(self):
        self.t+=1
    def update(self,layer):
        #for dense layer gradient is dw and db (dw,db)
        gradient=(layer.dw,layer.db)
        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum=np.zeros_like(layer.w)
            layer.weight_variance=np.zeros_like(layer.w)

            layer.bias_momentum=np.zeros_like(layer.b)
            layer.bias_variance=np.zeros_like(layer.b)
        layer.weight_momentum=self.beta1*layer.weight_momentum+(1-self.beta1)*gradient[0]
        layer.weight_variance=self.beta2*layer.weight_variance+(1-self.beta2)*gradient[0]**2

        layer.bias_momentum=self.beta1*layer.bias_momentum+(1-self.beta1)*gradient[1]
        layer.bias_variance=self.beta2*layer.bias_variance+(1-self.beta2)*gradient[1]**2

        
        weight_momentum_unbaised=layer.weight_momentum/(1-(self.beta1**self.t))
        weight_variance_unbaised=layer.weight_variance/(1-(self.beta2**self.t))

        bias_momentum_unbaised=layer.bias_momentum/(1-(self.beta1**self.t))
        bias_variance_unbaised=layer.bias_variance/(1-(self.beta2**self.t))


        layer.w-=self.lr*weight_momentum_unbaised/(np.sqrt(weight_variance_unbaised)+self.epsilon)            
        layer.b-=self.lr*bias_momentum_unbaised/(np.sqrt(bias_variance_unbaised)+self.epsilon)


class Gradient_Decent():
    def __init__(self,lr=0.01):
        self.lr=lr

    def __name__(self):
        return "Gradient_Decent"

    def pre_update(self):
        pass
    def update(self,layer):
        gradient=(layer.dw,layer.db)
        layer.w-=self.lr*gradient[0]
        layer.b-=self.lr*gradient[1]


class Learning_Rate_Scheduler():
    def __init__(self,optimizer,):
        self.optimizer=optimizer
        
    def constant(self,epoch):
        self.optimizer.lr=self.optimizer.lr

    def multiplicative_decay(self,epoch,power=0.95):
        self.optimizer.lr=self.optimizer.lr*power

    def step_decay(self,epoch,drop=0.5,epochs_drop=10):
        self.optimizer.lr=self.optimizer.lr*drop**(epoch//epochs_drop)
    
    def cyclic(self,epoch,base_lr=0.001,max_lr=0.010,step_size=2000):
        cycle=np.floor(1+epoch/(2*step_size))
        x=np.abs(epoch/step_size-2*cycle+1)
        lr=base_lr+(max_lr-base_lr)*np.maximum(0,(1-x))
        self.optimizer.lr=lr
    