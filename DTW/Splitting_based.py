import random
import h5py
from ExactS import ExactS
from distance import Distance, pop_random

#random.seed (0)

def heuristic_suffix_opt(traj_c, traj_q, index, opt, DIS_R):
    if index == len(traj_c):
        return 999999
    if opt == 'POS' or opt == 'POS-D':
        return 999999
    if opt == 'PSS':
        return DIS_R.DTW(traj_c[index:][::-1],traj_q[::-1])

def heuristic(traj_c, traj_q, opt, delay_K=5):
    delay = 0
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    split_point = 0
    DIS = Distance(len(traj_c), len(traj_q))
    DIS_R = Distance(len(traj_c), len(traj_q))
    pos_d_coll = []
    pos_d_f = False
    temp = 'non'
    if opt != 'POS-D':
        for i in range(len(traj_c)):
            #submit prefix
            presim = DIS.DTW(traj_c[split_point:i+1],traj_q)
            sufsim = heuristic_suffix_opt(traj_c, traj_q, i+1, opt, DIS_R)
            #print('-> maintain:', subtraj, subsim)
            #print('prefix:', [split_point, i], presim)
            #print('suffix:', [i+1, len(traj_c)-1], sufsim)S
            if presim < subsim or sufsim < subsim:
                temp = i + 1
                subsim = min(presim, sufsim)
                if presim < sufsim:
                    subtraj = [split_point, (temp-1)]
                else:
                    subtraj = [temp, len(traj_c)-1]
                split_point = temp
                DIS = Distance(len(traj_c[i:]), len(traj_q))
    else:
        i = -1
        while True:
            i = i + 1
            if temp != 'non':
                i = temp
                temp = 'non'
            if i == len(traj_c) - 1:
                break
            #submit prefix
            presim = DIS.DTW(traj_c[split_point:i+1],traj_q)
            if pos_d_f == False and presim < subsim: #open delay
                pos_d_f = True
            if pos_d_f == True and delay < delay_K:
                delay = delay + 1
                pos_d_coll.append((presim, i))
                continue
            if pos_d_f == True and delay == delay_K:
                sort = sorted(pos_d_coll, key=lambda d: d[0])
                temp = sort[0][1] + 1
                subsim = sort[0][0]
                subtraj = [split_point, (temp-1)]
                split_point = temp
                DIS = Distance(len(traj_c[sort[0][1]:]), len(traj_q))
                delay = 0
                pos_d_f = False
                pos_d_coll = []
        if subsim == 999999: #for extreme cases
            if pos_d_coll == []:
                presim = DIS.DTW(traj_c[split_point:i+1],traj_q)
                pos_d_coll.append((presim, i))
            sort = sorted(pos_d_coll, key=lambda d: d[0])
            temp = sort[0][1] + 1
            subsim = sort[0][0]
            subtraj = [split_point, (temp-1)]
            
    return subsim, subtraj

if __name__ == '__main__':
    f = h5py.File('./data/porto_querydb.h5','r')     
    (cand, query) = pop_random(f['/db/num/'].value)

    traj_C = f['/db/trips/'+str(cand)].value
    traj_Q = f['/db/trips/'+str(query)].value    
    subsim, subtraj, subset = ExactS(traj_C, traj_Q)

    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim)
    
    '''
    POS: maintain prefix only O(n)
    POS-D: maintain prefix and delay k steps only O(n)
    PSS: backward-train-model for suffix traj O(n)
    '''
    opt = 'PSS' #POS, POS-D, PSS
    ap_subsim, ap_subtraj = heuristic(traj_C, traj_Q, opt,  delay_K=5)
    
    print('ap-sub-trajectory', ap_subtraj)
    print('ap-sub-similarity', ap_subsim)
    print('approximate ratio', ap_subsim/subsim)