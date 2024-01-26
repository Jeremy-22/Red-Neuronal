#importamos las librerias
import random
import numpy as np

class Network(object): #aquí se define la clase del objeto,en donde
                       #tiene como herencia su argumento.
    def __init__(self, sizes): #Definimos el atributo init como privado
                               #al cual solo se pueden acceder a los métodos 
                               #privados por otros métodos dentro de la misma clase
                               #con argumento self y sizes, donde self es una varible
                               #proporcionada por python y sizes las caps.
        self.num_layers = len(sizes) #Aquí se define la varible (self.num_layers), la cual
                                     #es el número de elementos de sizes, es decir, el
                                     #numero de capas, dentro de self
        self.sizes = sizes #es un atributo publico del objeto self
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
                    #Se genera en la lista de capas una matriz de entradas aleatorias, en donde
                    #donde cada matriz tiene "y" filas y una columna, además "y" es el numero de
                    #neuronas en cada capa sin contar la primera
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
                    #Se inician los pesos de la red de manera aleatoria, la cual es de forma
                    #similar al de los sesgos, donde nos da una matriz con "y" filas y columnas
                    #"x", además zip se utiliza para combinar dos listas en una sola lista de tuplas
    def feedforward(self, a):
        #definimos feedforward para poder evaluar la red, es decir, realiza la propagación hacia 
        #adelante en la red neuronal. Toma una entrada a y pasa a través de las capas de la red
        #aplicando una función de activación (en este caso, la función sigmoide) en cada capa.    
        for b, w in zip(self.biases, self.weights):
            a = Network.sigmoid(np.dot(w, a)+b)
        return a
        #aquí definimos la variable "a" (función de activación)
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): 
        #Definimos el Stochastic Gradient Descent, con argumento, los datos de prendizaje, el número
        #de epocas, el mini_batch_size, la taza de aprendizaje.
        training_data = list(training_data)
        #transforma los datos de aprendizaje en listas o tuplas
        n = len(training_data)
        #"n" guarda el número de elementos de "training_data"
        if test_data: #se verifica si se proporcionan datos de prueba
            test_data = list(test_data) #transforma los datos de prueba en listas
            n_test = len(test_data) #almacena el número de elementos de la variable "test_data"
        for j in range(epochs): #Se realiza un bucle para iterar sobre el numero de epocas que 
                                #se deasea entrenar.
            random.shuffle(training_data) #de forma aleatoria ordena las entradas de la lista.
            mini_batches = [
                training_data[k:k+mini_batch_size] # Divide los datos de entrenamiento en minibatch.
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:#Inicia un bucle que itera a través de cada minibatch
                self.update_mini_batch(mini_batch, eta)#Actualiza los pesos y sesgos de la red utilizando 
                                                       #el minibatch actual y la tasa de aprendizaje.
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                #si se proporcionan los datos de prueba, entonces se imprime el número de época actual, 
                #el número de predicciones correctas en los datos de prueba y el número total de ejemplos de prueba.
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        #para cada mini_batch se aplica un único paso de descenso del gradiente
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Inicializa una lista nabla_b con gradientes de biases, pero todos los elementos se inicializan con 
        #matrices de ceros.
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # Aquí x es la entrada de la red, y y es el dato real que queremos que nos 
            #de ademas de actualizar los pesos y sesgos de la red de acuerdo con una única iteración del 
            #descenso de gradiente, utilizando solo los datos de entrenamiento en mini_batch.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Llama al método backprop para calcular los gradientes del error con respecto a 
            #(delta_nabla_b) y con respecto a (delta_nabla_w)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #Aquí se actuliza la variable
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        #Se comienza mezclando aleatoriamente los datos de entrenamiento y luego los divide en 
            #minilotes del tamaño apropiado (el numero de elementos del mini_batch).
    def backprop(self, x, y): #esto nos devuelveuna tupla que representa el gradiente para la función de costo C_x
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #son listas de capa por capa o elemento a elemento de matrices similiraes
                                            
        # feedforward
        activation = x #Inicializa una variable activation con la entrada "x". Esta variable se usará 
                       #para mantener la activación de cada capa durante la propagación hacia adelante.
        activations = [x] #genera una lista que alamacena todas las activaciones capa por capa
        zs = [] # los mismo que arriba
        for b, w in zip(self.biases, self.weights): #se inicia el bucle que itera los pesos y segos
            z = np.dot(w, activation)+b #calculamos la entrada z de la capa actual 
            zs.append(z) #agrega la entrada z a la lisla zs 
            activation = Network.sigmoid(z) #La activación de la capa actual aplica la funcion de 
                                    #activasion sigmoide con la entrada z
            activations.append(activation) #agrega la activación a la lista activaciones

        delta = self.cost_derivative(activations[-1], y) * \
            Network.sigmoid_prime(zs[-1])  #Calcula el delta de error en la capa de salida
        nabla_b[-1] = delta #Almacena el gradiente de sesgos en la última capa (-1) en la lista nabla_b.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):#bucle que itera a través de las capas ocultas de la red neuronal
                           #l variable l es pequeña y l=1 es la ultima capa de la red
            z = zs[-l]     # y -1 es la penultima capa y así sucesivamente, z respresenta la entreda de la capa actual
                          
            sp = Network.sigmoid_prime(z) #es el grandiente de la función de activación aplicada a z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #representa el error en la capa actual,
                                   # y se calcula utilizando los pesos y el error en la capa siguiente
            nabla_b[-l] = delta # Almacena el gradiente de biases en la capa actual en la lista nabla_b.
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) #Devuelve los gradientes de sesgos y pesos calculados para todas las capas de la red.

    def evaluate(self, test_data): #definimos evaluate
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]#evalúa la red neuronal en cada ejemplo de test_data.
        return sum(int(x == y) for (x, y) in test_results)# devuelve l número de entradas de prueba 
                                   #para las cuales la red neuronal genera el resultado correcto.

    def cost_derivative(self, output_activations, y): #definimos la derivada de la función de costo
        return (output_activations-y) #esto nos devuelve el vector de derivadas parciales de la funcion
        #de costo con respecto a la salida de la red
    def sigmoid(z): #definimos la función de activación sigmoide con argumento z
        return 1.0/(1.0+np.exp(-z)) #nos devuelve la funcion de activación sigmoidal
    def sigmoid_prime(z): #definimos la derivada de la funcion sigmoidal
        return Network.sigmoid(z)*(1-Network.sigmoid(z)) #nos regresa la derivada de la función sigmoidal