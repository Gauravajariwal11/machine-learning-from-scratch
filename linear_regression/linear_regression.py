#name- Gaurav Ajariwal
#UTA ID- 1001396273

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# trainfile = sys.argv[1]
# testfile = sys.argv[2]

# print(train_data)

def linear_regression(training_file, degree, lamda, test_file):

    # train_data = np.loadtxt(training_file)
    # training_file = "./f1.txt"
    train_data = np.genfromtxt(training_file)           ######## TRAINING STAGE
    rows = train_data.shape[0]                          #checking no of rows
    columns = train_data.shape[1]                        #checking no of cols
    phi = []
    target = []
    # x = train_data[:,0]
    # print(x)

    for i in train_data:
        result = [1]
        for j in i[:-1]:
            for k in range(1, degree+1):
                result.append(float(j)**float(k))
    
        result = np.array(result)
        phi.append(result)
        target.append(i[-1])

    phi = np.array(phi)
    target = np.array([target]).T
    w = np.linalg.pinv(lamda * np.identity(len(phi[0])) + phi.T @ phi ) @ phi.T @ target

    for i in range(w.size):
        print('w%d=%.4f' %(i,w[i]))

    # test_file='./f1.txt'  
    print('\n This is testing phase.')

    test_data= np.genfromtxt(test_file)                         ###### TESTING STAGE
    phi_test=[]
    target_val=[]
    accuracy=0

    for i, i2 in enumerate(test_data):
        result = [1]
        for j in i2[:-1]:
            for k in range(1, degree+1):
                result.append(float(j)**float(k))
    
        result = np.array(result)
        phi_test.append(result)
        target_val.append(i2[-1])
        target_output=w.T @ phi_test[i]
        squared_error=(target_val[i]-target_output)**2
        print('ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' %( (i+1),target_output, target_val[i], squared_error))
    
        if(squared_error<1):
            accuracy+=1
    
    print('\n')
    print('Accuracy=%.2f'%(accuracy/len(test_data)))
    return

training_file=sys.argv[1]
degree=int(sys.argv[2])
lamda=int(sys.argv[3])              #cant put lambda name in python
testing_file=sys.argv[4]

linear_regression(training_file,degree,lamda, testing_file)