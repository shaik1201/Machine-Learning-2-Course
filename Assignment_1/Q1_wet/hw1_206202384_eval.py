from hw1_206202384_train import DNN
import torchvision.datasets as dsets
import numpy as np
import pickle

def evaluate_hw1():
    test_dataset = dsets.MNIST(root='./data/',
                              train=False, 
                              download=True)
    
    dnn = pickle.load(open('model_q1.pkl','rb'))
    accuracy = dnn.compute_accuracy(test_data=test_dataset, output_nodes=10)
    print(f'accuracy: {accuracy}')

evaluate_hw1()