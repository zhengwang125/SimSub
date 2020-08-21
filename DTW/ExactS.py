from distance import Distance, pop_random
import h5py
import random
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

#random.seed (0)

def ExactS(traj_c, traj_q):
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    subset = {}
    N = len(traj_c)
    for i in range(N):
        DIS = Distance(len(traj_c[i:]), len(traj_q))
        for j in range(i, N):
            temp = DIS.DTW(traj_c[i:j+1], traj_q)
            #temp = similaritymeasures.dtw(traj_c[i:j+1], traj_q)[0]
            #print('sub-range:', [i, j], temp)
            subset[(i, j)] = temp
            if temp < subsim:
                subsim = temp
                subtraj = [i, j]
    return subsim, subtraj, subset

if __name__ == '__main__':
    f = h5py.File('./data/porto_querydb.h5','r')
#    print(f["/query/trips/1"].value)
#    print(f["/query/names/1"].value)
#    print(f["/db/trips/1"].value)
#    print(f["/db/names/1"].value)
#    print(f["/db/num"].value)
#    print(f["/query/num"].value)
    (cand, query) = pop_random(f['/db/num'].value)
    traj_C = f['/db/trips/'+str(cand)].value
    traj_Q = f['/db/trips/'+str(query)].value
    subsim, subtraj, subset = ExactS(traj_C, traj_Q)    
    #print('query:', traj_Q)
    #print('candidate:', traj_C)
    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim) 