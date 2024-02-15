import mnist_loader
import Network_2
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=Network_2([784, 30, 10]) 
#net.SG_momentum( training_data, 30, 10, 3.0,0.91, test_data=test_data)
net.SGD_momentum(training_data, 30, 10 ,3, 0.9, test_data=test_data)
#training_data, epochs, mini_batch_size, eta, momentum,test_data=None)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo

#archivo_lectura = open("red_prueba.pkl",'rb')
#net = pickle.load(archivo_lectura)
#archivo_lectura.close()

#net.SG( training_data, 10, 50, 0.5, test_data=test_data)

#archivo = open("red_prueba.pkl",'wb')
#pickle.dump(net,archivo)
#archivo.close()
#exit()