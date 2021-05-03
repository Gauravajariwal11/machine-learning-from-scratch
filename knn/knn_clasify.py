#Name- Gaurav Ajariwal
#UTA ID- 1001396273

import numpy as np
import sys

class PreprocessData:

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = []
        self.test_data = []
        self.dimensions = 0
    
    def load_data(self):

        train_file = open(self.train_file, "r")                     # read the train file
        train_lines = train_file.readlines()   

        test_file = open(self.test_file, "r")                       # read the test file
        test_lines = test_file.readlines()  

        self.dimensions = len(list(filter(None, train_lines[1].split(" "))))                            # num of dimensions

        self.train_data = np.zeros((1,self.dimensions-1))        
        self.test_data = np.zeros((1,self.dimensions-1))
        self.train_labels = np.zeros((1,1))                                         # train labels
        self.test_labels = np.zeros((1,1))                                          # test labels
        
        for line in train_lines:
            x = line.split(" ")
            x = list(filter(None, x))                                              
            x = list(map(float, x))        
            self.train_data = np.vstack((self.train_data, x[:-1]))
            self.train_labels = np.vstack((self.train_labels, x[-1]))

        for line in test_lines:
            x = line.split(" ")
            x = list(filter(None, x))  
            x = list(map(float, x))
            self.test_data = np.vstack((self.test_data, x[:-1]))
            self.test_labels = np.vstack((self.test_labels, x[-1]))

        self.train_data = self.train_data[1:]
        self.test_data = self.test_data[1:]  
        self.train_labels = self.train_labels[1:]
        self.test_labels = self.test_labels[1:] 
        
class KNN(PreprocessData):
    def __init__(self, train_file, test_file, k):
        PreprocessData.__init__(self, train_file, test_file)
        self.k = k
        self.count = 0

    def normalize(self):
        mean = np.mean(self.train_data, axis=0)                 #getting mean from np library
        std = np.std(self.train_data, axis=0)                   #getting std from np library
        for i in range(self.train_data.shape[1]):
            self.train_data[:,i] = (self.train_data[:,i]-mean[i])/std[i]
            self.test_data[:,i] = (self.test_data[:,i]-mean[i])/std[i]
    
            
    def euclidian(self, v1, v2):
        v1 = v1.reshape(1,v1.shape[0])
        eculid_dist = v2[:,] - v1
        eculid_dist = np.square(eculid_dist)
        eculid_dist = np.sum(eculid_dist, axis = 1)
        eculid_dist = np.sqrt(eculid_dist)
        return eculid_dist
    
    def results(self, y_predicted, object_id):
        true_class = self.test_labels[object_id]
        (values, counts) = np.unique(y_predicted, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        accuracy = 0
        if (true_class[0] == int(predicted_class)):
            accuracy = 1
            self.count += 1
        else:
            accuracy = 0
        print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format((object_id+1), int(predicted_class), int(true_class[0]), accuracy))

    def accuracy(self):
        accuracy = self.count/self.test_labels.shape[0]
        print("\nclassification accuracy={:6.4f}".format(accuracy))

    def knn_classify(self):
        test_data = self.test_data.shape[0]
        for i in range(test_data):
            euclidian_distance = self.euclidian(self.test_data[i,:], self.train_data)
            sorted_labels = self.train_labels[euclidian_distance[:].argsort()]
            self.results(sorted_labels[:self.k], i)
        self.accuracy()
        

def main():

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    k = int(sys.argv[3])

    knn = KNN(train_file, test_file, k)
    knn.load_data()
    knn.normalize()
    knn.knn_classify()

main()
