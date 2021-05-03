#Name- Gaurav Ajariwal
#UTA ID- 1001396273

import numpy as np
import tqdm
import itertools
import random
import math
import sys

def loadDataSet(filename):

    data = []
    with open(filename, 'r') as load_file:
        for line in load_file:
            if ' ' in line:
                data.append(tuple(map(float,line.split())))
            else:
                data.append(float(line.rstrip()))
    # print(data)
    return data

def distEclud(p0, p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)


def kmeans(data_file, k, initialization):

    datapoints = loadDataSet(data_file)
    k = int(k)
    # d - Dimensionality of Datapoints
    d = len(datapoints[0])

    data1 = np.array(datapoints).astype(float)
    rows = data1.shape[0]
    columns = data1.shape[1]
    
    #Limit our iterations
    Max_Iterations = 1000
    i = 0
    
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    #Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        cluster_centers += [random.choice(datapoints)]
        

        #in case of random points being chosen randomly
        force_recalculation = False     
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        #Update Point's Cluster Alligiance
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            
            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):
                
                dist = distEclud(datapoints[p],cluster_centers[c])
                
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c   # Reassign Point to new Cluster
        
        
        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                
                #If our initial random assignment was poorly chosen
                else: 
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    
            
            cluster_centers[k] = new_center
    
        
    # print(list(datapoints))
    # print(type(datapoints))
    # print(type(datapoints[0]))
    # print(type(datapoints[0][1]))
    # print(datapoints)
    # print(datapoints[0][1])
    # print(datapoints[1])
    # print ("Assignments", [x+1 for x in cluster])

    cluster = [x+1 for x in cluster]
    for i in list(datapoints):
        # print(type(datapoints[i]))
        # print(i)
        if (columns==1):
            print('%10.4f --> cluster %d\n' %(datapoints, cluster))
        else:
            # print(i[0],i[1])
            print("({:.4f}, {:.4f}) --> cluster {}\n".format(*i, cluster))
                
            
data_file = sys.argv[1]
k = sys.argv[2]
initialization = sys.argv[3]

kmeans(data_file, k, None)

