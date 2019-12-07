import math
import random
import numpy as np
import functions as func
import os
import pandas as pd
import re
from pprint import pprint


DEFAULT_HIDDEN_LAYER_SIZE=4
OUTPUT_LAYER_SIZE=1
INPUTS=5
NUMBER_OF_ACTIONS=2

class Network:
    def __init__(self, shape):
        self.L=len(shape)
        self.W=[]
        self.Theta=[]
        for l in range(self.L-1):
            self.W.append(np.random.normal(loc=0,scale=1/np.sqrt(shape[l]),
                                       size=(shape[l+1],shape[l])))
            self.Theta.append( np.zeros(shape = (shape[l+1]) ) )

    def __eq__(self, network):
        if self.L != network.L:
            return False

        for w1, w2 in zip(self.W, network.W):
            if w1.shape != w2.shape or not (w1==w2).all():
                return False
        for t1, t2 in zip(self.Theta, network.Theta):
            if t1.shape != t2.shape or not (t1==t2).all():
                return False

        return True
            





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

    def save(self, path):
        """ 
        The network will be saved into csv-files in the directory provided by path. This directory
        will be added if it doesn't already exist.
        """
        dirs = path.split('/')
        for i in range(len(dirs)):
            p = '/'.join(dirs[:i+1])
            if not os.path.exists(p):
                os.mkdir(p)


        for i,w in enumerate(self.W):
            fname = '{}/w{}.csv'.format(path, i)
            df = pd.DataFrame(w)
            df.to_csv(fname, header = None, index = None)

        for i, theta in enumerate(self.Theta):
            if theta is not None:
                fname = '{}/t{}.csv'.format(path, i)
                df = pd.DataFrame(theta)
                df.to_csv(fname, header = None, index = None)

    def load(self, path):
        """
        Load a network from a directory provided by path. Files that will be read are on the
        format w{i}.csv for weights and t{i}.csv from thresholds.
        Example:
            -> network = Network(shape = [4, 4, 2])
            -> network.load(path = 'results/some_directory')
        """
        
        files = os.listdir(path)

        weight_files, theta_files = [], []
        for file in files:

            f = os.path.join(path, file)
            if re.match(r'w\d+.csv', file):
                weight_files.append(f)
            elif re.match(r't\d+.csv', file):
                theta_files.append(f)

        weight_files.sort()
        theta_files.sort()

        self.L = min(len(weight_files), len(theta_files)) + 1
        self.W = []
        self.Theta = []

        for i in range(self.L-1):
            w = np.genfromtxt(weight_files[i], delimiter=",")
            t = np.genfromtxt(theta_files[i], delimiter=",")
            self.W.append(w)
            self.Theta.append(t)





if __name__ == "__main__":
    a = [4,4,2]
    x=Network(shape = a)
    x.print_network()
    for i in range(10):
        x.mutate(mutationrate=0.05, creeprate=0.4)

    x.print_network()



