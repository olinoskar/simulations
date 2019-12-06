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
            b = np.dot(self.W[l], V) - self.Theta[l]
            beta = 1
            V = 1./(1+np.exp(-b*beta))
        return V


    def feed(self, X):
        V = self.prop_forward(X)
        return V

    def mutate(self,mutationrate,creeprate):
        for w in range(len(self.W)):
            self.W[w]=self.W[w]+creeprate*(2*np.random.random(size=self.W[w].shape)-1)*np.random.choice([1,0], size=self.W[w].shape,p=[mutationrate, 1-mutationrate])
        for t in range(len(self.Theta)):
            self.Theta[t]=self.Theta[t]+creeprate*(2*np.random.random(size=self.Theta[t].shape)-1)*np.random.choice([1,0], size=self.Theta[t].shape,p=[mutationrate, 1-mutationrate])

        return


if __name__ == "__main__":
    a = [4,4,2]
    x=Network(shape = a)
    x.print_network()
    for i in range(10):
        x.mutate(mutationrate=0.05, creeprate=0.4)

    x.print_network()



