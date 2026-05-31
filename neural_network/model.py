import numpy as np
import matplotlib.pyplot as plt

import sys
import time

def progress_bar(current, total, bar_length=40):
    fraction = current / total
    arrow = int(fraction * bar_length) * '█'
    padding = (bar_length - len(arrow)) * '-'
    ending = '\n' if current == total else ''
    
    # \r moves the cursor back to the start of the line
    sys.stdout.write(f'\rProgress: |{arrow}{padding}| {int(fraction * 100)}%')
    sys.stdout.flush()
    if current == total:
        print()

class Sequential():
    def __init__(self):
        self.layers=[]
    def add(self,layer):
        self.layers.append(layer)
    def forward(self,x):
        for i in self.layers:
            x=i.forward(x)
        return x
    def backward(self,gradient):
        for i in reversed(self.layers):
            gradient=i.backward(gradient)
        return gradient
    def complie(self,loss,optimizer):
        self.loss=loss
        self.optimizer=optimizer
        for i in self.layers:
            if hasattr(i,'w'):
                i.optimizer=optimizer
        
    def fit(self,x,y,epochs,lr,batch_size=32):
        losses=[]

        for epoch in range(epochs):
            index=np.arange(x.shape[0])  
            epoch_loss=0
            for i in range(0,x.shape[0],batch_size):
                
                self.optimizer.pre_update()
                batch_index=index[i:i+batch_size]
                x_batch=x[batch_index]
                y_batch=y[batch_index]

                output=self.forward(x_batch)
                Loss=self.loss.forward(output,y_batch)
                gradient=self.loss.backward(output,y_batch)
                self.backward(gradient)
                epoch_loss += Loss
            
            losses.append(epoch_loss/(x.shape[0]//batch_size))

            if epoch%10==0:
                progress_bar(epoch, epochs)
                print(f' Epoch {epoch}, Loss: {epoch_loss/(x.shape[0]//batch_size)}')
        
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()

        plt.show()
    def predict(self,x):
        output=self.forward(x)
        return output
