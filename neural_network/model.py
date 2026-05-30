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
        
    def fit(self,x,y,epochs,lr):
        losses=[]
        for epoch in range(epochs):
            self.optimizer.pre_update()
            output=self.forward(x)
            Loss=self.loss.forward(output,y)
            losses.append(Loss)
            gradient=self.loss.backward(output,y)
            self.backward(gradient)
            plt.plot(losses)

            if epoch%100==0:
                progress_bar(epoch, epochs)
                print(f' Epoch {epoch}, Loss: {Loss}')
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()

        plt.show()
    def predict(self,x):
        output=self.forward(x)
        return output
