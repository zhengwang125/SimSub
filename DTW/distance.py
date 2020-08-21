import numpy as np
from numba import jit
import random

def pop_random(lst):
    idx1 = random.randrange(1, lst)
    idx2 = random.randrange(1, lst)
    return (idx1,idx2)

class Distance:
    def __init__(self, N, M): # N = length of C, M = length of Q
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0,1:] = np.inf
        self.D0[1:,0] = np.inf
        self.D = self.D0[1:,1:] #shallow copy!!
        #print(self.D)
    #@jit
    def DTW(self, traj_C, traj_Q, skip=[]):
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            if list(traj_C[i]) in skip:
                if self.flag[i,0] == 0:
                    self.D[i,:] = self.D[i-1,:]
                    self.flag[i,:] = 1
                continue
            for j in range(m):
                if self.flag[i,j] == 0:
                    cost =  np.linalg.norm(traj_C[i] - traj_Q[j])
                    self.D[i,j] = cost + min(self.D0[i,j],self.D0[i,j+1],self.D0[i+1,j])
                    self.flag[i,j] = 1
                    #print(self.D)
        return self.D[n-1, m-1]
