import random
import numpy as np
class Network(object):
    def __init__(self, sizes): 
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, a):   
        for b, w in zip(self.biases, self.weights):
            a = Network.sigmoid(np.dot(w, a)+b)
        return a
    def SGD_momentum(self, training_data, epochs, mini_batch_size, eta, momentum,
            test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data) 
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_momentum(mini_batch, eta, momentum)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
    def update_mini_batch_momentum(self, mini_batch, eta, momentum):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        v_w = [np.zeros(w.shape) for w in self.weights]
        v_b = [np.zeros(b.shape) for b in self.biases]

        v_b = [momentum * vb - nb
                      for vb, nb in zip(v_b, nabla_b)]
        v_w = [momentum * vw - nw
                      for vw, nw in zip(v_w, nabla_w)]
        self.biases = [b - (eta / len(mini_batch))*vb for b, vb in zip(self.biases, v_b)]
        self.weights = [w - (eta / len(mini_batch))*vw for w, vw in zip(self.weights, v_w)]
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
                                            
        # feedforward
        activation = x 
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) 
            activation = Network.sigmoid(z)
            activations.append(activation)

        delta = self.binary_cross_entropy_derivative(activations[-1], y) * \
              Network.sigmoid_prime(zs[-1]) #La implementamos sobre backprop
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l] 
                          
            sp = Network.sigmoid_prime(z) 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) 
    def evaluate(self, test_data):  
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    def binary_cross_entropy_loss(self, output_activations, y):#definimos la funcion binary cross-entroy
        output_activations = np.clip(output_activations, 1e-15, 1 - 1e-15)# para que vaya de 0 o 1
        loss = -np.mean((y * np.log(output_activations) + (1 - y) * np.log(1 - output_activations)))
        return loss
    def binary_cross_entropy_derivative(self, output_activations, y):
        epsilon = 1e-15  # Para evitar divisiones por cero
        y = np.clip(output_activations, epsilon, 1 - epsilon)  # Clip para evitar log(0)
        derivative = -((y / output_activations) - ((1 - y) / (1 - output_activations)))
        return derivative
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z)) 
    def sigmoid_prime(z): 
        return Network.sigmoid(z)*(1-Network.sigmoid(z)) 