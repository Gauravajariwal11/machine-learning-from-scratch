#Name- Gaurav Ajariwal
#UTA ID- 1001396273

import os
import sys
import pandas as pd
import numpy as np
import math
import random

weight_matrix = []
each_layer_error = []
each_layer_output = []
l_with_bias = []
classes = []
num_classes = 0
predicted_label = []

def normalize_data(data):
    max = np.amax(data)
    data=data/max
    return data

def bias(data): 
    x=np.ones((1, data.shape[0]))
    bias=np.concatenate((x, data), axis = 1)   
    return bias

def sigmoid(data): 
    return (1/(1+np.exp(-data)))

def one_hot_encoding(train_labels):                #to encode the last column of training data
    one_hot_encoding_label = np.zeros((1, num_classes))
    index = np.where(classes == train_labels)
    one_hot_encoding_label[0][index] = 1
    return one_hot_encoding_label


def initialize_weight_matrix(units_per_layer, layers, num_features, num_classes): 
    np.random.seed(42)
    input_weights = initialize_weights(units_per_layer, num_features + 1)
    weight_matrix.append(input_weights)

    for i in range(1, layers): 
        hidden_layer_weights = initialize_weights(units_per_layer, units_per_layer + 1)
        weight_matrix.append(hidden_layer_weights)

    if(layers != 0): 
        output_weights = initialize_weights( num_classes, units_per_layer + 1)
        weight_matrix.append(output_weights)
        

def initialize_weights(x,y):
       return np.array(np.random.uniform(low= -0.05, high= 0.05, size=(x, y)))

def train(trainData, layers): 
    trainData = bias(trainData)
    each_layer_output.clear()
    l_with_bias.clear()
    each_layer_output.append(trainData)
    l_with_bias.append(trainData)

    for i in range(layers+1): 
        trainData = np.dot(trainData, weight_matrix[i].T)
        trainData = sigmoid(trainData)
        each_layer_output.append(trainData)

        if(i != layers):
            trainData = bias(trainData)

        l_with_bias.append(trainData)
        
    return trainData


def is_valid_class(predicted_label, test_labels, max_value):
    for i in range(len(predicted_label)): 
        if(max_value == predicted_label[i] and predicted_label[i] == test_labels): 
            return True
    return False

def accuracy(predicted_label, testing_labels):                  #get the accuracy value with predicted label vs testing label
    test_labels = []  
    test_accuracy = []  
    acc_value = 0

    for ID in range(len(predicted_label)):
        max_val = np.amax(predicted_label[ID])
        max_count = np.count_nonzero(predicted_label[ID] == max_val)        

        if(max_count == 1): 
            indexLabel = np.argmax(predicted_label[ID]) 
            label =  classes[indexLabel]
          
            if(label == testing_labels[ID]): 
                acc_value += 1
        else: 
            if(is_valid_class(predicted_label[ID], testing_labels[ID],max_val)): 
                labelIndex = np.argmax(predicted_label[ID]) 
                label = classes[random.choice(labelIndex)]
                acc_value += 1/max_count
    
        test_labels.append(label)  
        
        test_accuracy.append(acc_value/len(testing_labels)*100)

        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n'% (ID+1, label, testing_labels[ID], test_accuracy[ID]))   
    print("classification accuracy=%6.4f\n" %(acc_value/len(testing_labels)))



def updateWeights_CalErrors(data, training_labels, layers, learning_rate):

    each_layer_error.clear()
    onehotencode = one_hot_encoding(training_labels)
    term1 = (data - onehotencode)
    term2 = (1-data) * data
    each_layer_error.append(term1 * term2)
    
    for i in range(layers, 0, -1):                                                      #hidden layers 
        weight_without_bias = np.delete(weight_matrix[i],0 ,axis=1)
        del_u = (np.dot(each_layer_error[0], weight_without_bias )) 
        del_i = del_u * (each_layer_output[i] * (1 - each_layer_output[i]))
        each_layer_error.insert(0, del_i)
       

    for i in range(len(weight_matrix)):       
        weight_matrix[i] = weight_matrix[i] - (learning_rate * (np.dot(np.array(each_layer_error[i]).T, l_with_bias[i])))

def main(): 
    #take inputs here... 6 when we want to include units per hidden layer, and 5 for when no hidden layer. 
    if(len(sys.argv) == 6 or len(sys.argv) == 5): 

        if(len(sys.argv) == 6):
            train_file = sys.argv[1]
            test_file = sys.argv[2]
            layers = int(sys.argv[3]) - 2 
            units_per_layer = int(sys.argv[4])
            rounds = int(sys.argv[5])
        else: 
            train_file = sys.argv[1]
            test_file = sys.argv[2]
            layers = int(sys.argv[3]) - 2 
            units_per_layer = 0
            rounds = int(sys.argv[4])
        
        # print(len(sys.argv)) making sure about sys.argv len
        train_data = np.loadtxt(train_file)          #Get the training file
        test_data = np.loadtxt(test_file)            #get the test file
        
        training_labels = train_data[:, -1]           #get the last column of train data
        train_data = normalize_data(train_data[:, :-1])         #Normalize the train data

        test_labels = test_data[:, -1]
        test_data = normalize_data(test_data[:, :-1])

        global classes
        global num_classes
        classes = np.unique(training_labels)
        num_classes = len(classes)

        if(len(sys.argv) == 5):
             units_per_layer = num_classes
               
        initialize_weight_matrix(units_per_layer, layers, train_data.shape[1], num_classes)    
       
        learning_rate = 1
        for i in range(rounds) : 
                    
            for index in range(len(train_data)): 
                data = train_data[index, np.newaxis]                 
                data = train(data, layers)

                updateWeights_CalErrors(data, training_labels[index], layers, learning_rate)
                
                
            learning_rate *= 0.98

        for index in range(len(test_data)): 
            data = test_data[index, np.newaxis]                 
            data = train(data, layers)
            predicted_label.append(data)

        accuracy(predicted_label, test_labels)

    else:
        print("Please provide valid command line inputs")


main()