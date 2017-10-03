import random
import numpy as np

# sigmoid squashing (value between 0 and 1)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#derivative of sigmoid
def sigprime(x):
    #assuming x is a result sigmoid(y)
    return x*(1-x)

class nnet(object):

    def __init__(self, sizes):
        #sizes is a list of arrays containing the number of neurons in that layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(35)
        self.biases = [2 * np.random.randn(y,1) - 1 for y in sizes[1:]]
        self.weights = [2 * np.random.rand(y,x) - 1 for x,y in zip(sizes[:-1],sizes[1:])]
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    #returns output with input a
    def feedfwd(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    #x is input, y is target
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedfwd
        activation = x
        activations = [x] #stores all activations layer by layer
        zs = [] #stores all z vectors, layer by layer
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward
        delta = self.cost_derivative(activations[-1],y) * sigprime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2,self.num_layers):
            z = zs[-1]
            sp = sigprime(z)
            print("delta\n",delta,len(delta))
            print("weights\n",self.weights[-l+1].transpose(),len(self.weights[-l+1].transpose()))
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases,nabla_b)]

    def cost_derivative(self,output_activations, y):
        return (output_activations-y)