import pandas as pd
import numpy as np


# Computation packages
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


#cost matrix: Euclidean distance
def comp_euclid_dist_matrix(x, y) -> np.array:
    dist = np.zeros((len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j] - y[i])**2
    #print(dist)
    return dist

def comp_accum_cost_matrix(x, y) -> np.array:
    distances =  comp_euclid_dist_matrix(x, y)

    #Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]
    #print("cost after for loop row" "\n", cost)
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]
    #print("cost after for loop column" "\n", cost)
    #Accumulate warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(cost[i-1, j],
                             cost[i,j-1],
                             cost[i-1, j-1]) + distances[i,j]
    cost = np.flipud(cost)
    return cost

def warp_path(cost):
    max_row = cost.shape[0]
    col = cost.shape[1] - 1
    row = 0
    path =[cost[0,col]]
    while (col > 1):
        while (row < max_row - 1):
            path.append(min(cost[row+1, col],
                             cost[row,col-1],
                             cost[row+1, col-1]))
            min_idx = np.argmin((cost[row+1, col],
                                 cost[row,col-1],
                                 cost[row+1, col-1]))
            if (min_idx == 0):
                row += 1
            if (min_idx == 1):
                col -= 1
            if (min_idx == 2):
                row += 1 
                col -= 1
    return(path)



x = [3, 1, 2, 2, 1]
y = [2, 0, 0, 3, 3, 1, 0]

cost_matrix = comp_accum_cost_matrix(x, y)
print (cost_matrix)
path = warp_path(cost_matrix)
print (path)
print(np.sum(path))