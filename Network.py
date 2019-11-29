import math
import random
import numpy as np
import functions as func


DEFAULT_HIDDEN_LAYER_SIZE=4
OUTPUT_LAYER_SIZE=1
INPUTS=5
NUMBER_OF_ACTIONS=2

class Network:
    def __init__(self,shape):
        self.L=len(shape)
        self.W=[]
        self.Theta=[]
        for l in range(self.L-1):
            #print("shape")
            #print(shape[l])
            self.W.append(np.random.normal(loc=0,scale=1/np.sqrt(shape[l]),
                                       size=(shape[l+1],shape[l])))
            self.Theta.append( np.zeros(shape = (shape[l+1]) ) )

    def print_network(self):
        print('\n\n--------------')
        for weights, thresholds in zip(self.W, self.Theta):
            print(weights)
            print(thresholds)
            print('--------------')
        print('\n\n')

    def prop_forward(self, X):
        V = X
            
        for l in range(self.L-1):
            #print(self.W[l])
            #print(V)
            #print(self.Theta[l])
            print("V innan =",V)
            print("W[l]",self.W[l])
            b = np.dot(self.W[l], V) - self.Theta[l]

            #V=func.sigmoid(b)
            V = b
            print("V efter =",V)
        return V


    def feed(self, X):
        V = self.prop_forward(X)
        return V
