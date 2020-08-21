from distance import Distance, pop_random
import h5py
import random
from ExactS import ExactS
#import similaritymeasures

#random.seed(0)
'''
# Three 2-D Trajectory Example
traj_C = np.array([[-122.39534, 37.77678],[-122.3992 , 37.77631],[-122.40235, 37.77594],[-122.40553, 37.77848],
               [-122.40801, 37.78043],[-122.40837, 37.78066]])
traj_Q = np.array([[-122.39472, 37.77672],[-122.3946 , 37.77679],[-122.39314, 37.77846]])
DIS = Distance(len(traj_C), len(traj_Q)) #init the matrix size |C|*|Q|
sp =  0
for i in range(len(traj_C)):
    print('-----' +str(i)+'-----')
    print(DIS.DTW(traj_C[sp:i+1], traj_Q)) #input prefix traj of traj_C
    print(DIS.DTW(traj_C[sp:i+1][::-1], traj_Q[::-1]))
'''

def SizeS(traj_c, traj_q, par=5):
    L = len(traj_q)
    L_lo = min(len(traj_c), int((L - par)))
    L_up = min(len(traj_c), int((L + par)))
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    N = len(traj_c)
    for i in range(N):
        DIS = Distance(len(traj_c[i:]), len(traj_q))
        for j in range(i, N):
            if (j - i + 1) <  L_lo or (j - i + 1) > L_up:
                 continue
            temp = DIS.DTW(traj_c[i:j+1], traj_q)
            #temp = similaritymeasures.dtw(traj_c[i:j+1], traj_q)[0]
            #print('sub-range:', [i, j], temp)
            if temp < subsim:
                subsim = temp
                subtraj = [i, j]
    return subsim, subtraj

if __name__ == '__main__':
    f = h5py.File('./data/porto_querydb.h5','r')
#    print(f["/query/trips/1"].value)
#    print(f["/query/names/1"].value)
#    print(f["/db/trips/1"].value)
#    print(f["/db/names/1"].value)
#    print(f["/db/num"].value)
#    print(f["/query/num"].value)
        
    (cand, query) = pop_random( f['/db/num/'].value)
    traj_C = f['/db/trips/'+str(cand)].value
    traj_Q = f['/db/trips/'+str(query)].value
    subsim, subtraj, subset = ExactS(traj_C, traj_Q)
    par = 5
    ap_subsim, ap_subtraj = SizeS(traj_C, traj_Q, par)
    #print('query:', traj_Q)
    print('sub-trajectory', ap_subtraj)
    print('sub-similarity', ap_subsim)
    print('approximate ratio', ap_subsim/subsim)