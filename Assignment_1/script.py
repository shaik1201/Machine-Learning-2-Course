import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import pickle
import time

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001


train_dataset = dsets.MNIST(root='./data/',
                               train=True, 
                               download=True)

test_dataset = dsets.MNIST(root='./data/',
                              train=False, 
                              download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)




class DNN:
    def __init__(self, sizes, epochs, lr):
        self.accuracy = []
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        self.params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

#         self.params = pickle.load(open('weights2.pkl','rb'))
    def relu(self, x, derivative=False):
        if derivative:
            return np.array([1 if i > 0 else 0 for i in x])
#             return np.heaviside(x, 0) 
        return np.array([max(i, 0) for i in x])
#         return max(0.0, x)
        
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
      # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
#         print(f'W1 SHAPE: {params["W1"].shape}')
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.relu(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.relu(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):

        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.relu(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.relu(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):

        for key, value in changes_to_w.items():
            self.params[key] -= self.lr * value

    def compute_accuracy(self, test_data, output_nodes):
        
        predictions = []

        for x in test_data:
            # scale and shift the inputs
            X_test, y_test = x
            inputs = (np.asfarray(X_test) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(y_test)] = 0.99
            output = self.forward_pass(np.array(inputs).flatten())
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(targets))

        return np.mean(predictions)

    def train(self, train_list, test_list, output_nodes):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x in train_list:
                X_train, y_train = x
                # scale and shift the inputs
                inputs = (np.asfarray(X_train) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(output_nodes) + 0.01
                targets[int(y_train)] = 0.99
                output = self.forward_pass(np.array(inputs).flatten())
                changes_to_w = self.backward_pass(targets, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(test_list, output_nodes)
            self.accuracy.append(accuracy)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(iteration+1, time.time() - start_time, accuracy * 100
    ))
            