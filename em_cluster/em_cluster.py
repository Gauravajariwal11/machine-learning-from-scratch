#Name- Gaurav Ajariwal
#UTA id- 1001396273

import sys
import numpy as np

def em_cluster(data_file, k, initialization, iterations):
    data = []
    k = int(k)
    iterations = int(iterations)

    with open(data_file, 'r') as file:
        for line in file:
            if ',' in line:
                data.append(line.split(', ')[0:-1])
            else:
                data.append(line.split()[0:-1])
    
    # Initialization
    data = np.array(data).astype(float)
    rows = data.shape[0]
    columns = data.shape[1]


    p = np.zeros((k, rows))

    #mean and weight
    mean = [[0 for i in range(columns)] for j in range(k)]
    weight = [0 for i in range(k)]

    # Covariance matrix array
    std = [[[0 for i in range(columns)] for j in range(columns)] for k in range(k)]

    # Randomly assign each point to a cluster
    for j in range(rows):
        p[np.random.randint(low=0, high=k)][j] = 1

    for _ in range(iterations):

        for i in range(k):
            temp = 0
            temp2 = 0
            
            # Mean calculation
            for j in range(rows): 
                temp += (p[i][j] * data[j])
            mean[i] = temp / sum(p[i])
            
            # Weight calculation
            weight[i] = sum(p[i]) / sum(sum(p))

            # Standard deviation calculation - covariance matrix
            for r in range(columns):
                for c in range(columns):
                    for j in range(rows):
                        temp2 += p[i][j] * (data[j][r] - mean[i][r]) * (data[j][c] - mean[i][c])
                    if (temp2 / sum(p[i]) < 0.0001 and r == c):
                        std[i][r][c] = 0.0001
                    else:
                        std[i][r][c] = temp2 / sum(p[i])

        pxj = 0
        n = 0

        for i in range(k):
            for j in range(rows):
                for l in range(k):
                    pxj += (1 / np.sqrt((2 * np.pi)**columns * np.linalg.det(std[l]))) * np.exp(-1/2 * (data[j] - mean[l]).T @ np.linalg.pinv(std[l]) @ (data[j] - mean[l])) * weight[l]
                n += (1 / np.sqrt((2 * np.pi)**columns * np.linalg.det(std[i]))) * np.exp(-1/2 * (data[j] - mean[i]).T @ np.linalg.pinv(std[i]) @ (data[j] - mean[i])) * weight[i]
                p[i][j] = n * weight[i] / pxj
                
        # Output
        print('After iteration %d:' % (_ + 1))
        for i in range(k):
                print('weight %d = %.4f, mean %d = (' % (i + 1, weight[i], i + 1), end = '')
                for j in range(columns):
                    print('%.4f' % mean[i][j], end = '')
                    if (j != columns - 1):
                        print(', ', end='')
                    else:
                        print(')')

    print('After final iteration:')
    for i in range(k):
        print('weight %d = %.4f, mean %d = (' % (i + 1, weight[i], i + 1), end = '')
        for j in range(columns):
            print('%.4f' % mean[i][j], end = '')
            if (j != columns - 1):
                print(', ', end='')
            else:
                print(')')

        for r in range(columns):
            print('Sigma %d row %d = ' % (i + 1, r + 1), end = '')
            for c in range(columns):
                print('%.4f' % std[i][r][c], end = '')
                if (c != columns - 1):
                    print(', ', end='')
                else:
                    print()


data_file = sys.argv[1]
k = sys.argv[2]
iterations = sys.argv[3]

em_cluster(data_file, k, None, iterations)