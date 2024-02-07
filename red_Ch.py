import random
import numpy as np

class Network(object):
    def __init__(self, sizes): 
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):   
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD_momentum(self, training_data, epochs, mini_batch_size, eta, momentum, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data) 
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_momentum(mini_batch, eta, momentum)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch_momentum(self, mini_batch, eta, momentum):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        v_b = [momentum * vb - (eta / len(mini_batch)) * nb for vb, nb in zip(self.v_b, nabla_b)]
        v_w = [momentum * vw - (eta / len(mini_batch)) * nw for vw, nw in zip(self.v_w, nabla_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, v_b)]
        self.weights = [w + vw for w, vw in zip(self.weights, v_w)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x 
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z) 
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.binary_cross_entropy(activations[-1], y) * self.sigmoid_prime(zs[-1]) 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l] 
            sp = self.sigmoid_prime(z) 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) 

    def evaluate(self, test_data):  
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def binary_cross_entropy(self, output_activations, y):
        output_activations = np.clip(output_activations, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(output_activations) + (1 - y) * np.log(1 - output_activations))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z): 
        return self.sigmoid(z) * (1 - self.sigmoid(z))
