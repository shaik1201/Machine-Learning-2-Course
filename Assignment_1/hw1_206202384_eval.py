from script import DNN
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import pickle

def evaluate_hw1():
    test_dataset = dsets.MNIST(root='./data/',
                              train=False, 
                              download=True)
    
    dnn = DNN(sizes=[784, 128, 64, 10], epochs=5, lr=0.001)
    dnn.params = pickle.load(open('weights_q1.pkl','rb'))
    accuracy = dnn.compute_accuracy(test_data=test_dataset, output_nodes=10)
    print(f'accuracy: {accuracy}')