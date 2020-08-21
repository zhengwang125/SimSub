from distance import data_loading, pop_random, submit
import random
import numpy as np
import evaluate
from t2vec import args
from ExactS import ExactS

#random.seed(0)

args.checkpoint = "./data/best_model_porto.pt"
args.vocab_size = 18867 #18867(porto)
m0 = evaluate.model_init(args)

args.checkpoint = "./data/best_model_portoR.pt"
args.vocab_size = 18949 #18949(portoR)
m1 = evaluate.model_init(args)

def heuristic_suffix_opt(invQ, Q, traj_c, index, opt, each_step_b_c):
    if traj_c[index:] == []:
        return 999999
    if opt == 'POS' or opt == 'POS-D':
        return 999999
    if opt == 'PSS':
        return np.linalg.norm(invQ - each_step_b_c[0, len(traj_c)-index-1, :])

def heuristic(traj_c, traj_q, opt, delay_K=5):
    delay = 0
    _, each_step_f_q = submit(m0, traj_q)
    Q = each_step_f_q[0,-1,:]
    invQ = -1
    each_step_b_c = -1
    if opt == 'PSS':
        _, each_step_b_q = submit(m1, traj_q[::-1])
        invQ = each_step_b_q[0,-1,:]
        _, each_step_b_c = submit(m1, traj_c[::-1])
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    split_point = 0
    h0 = None
    pos_d_coll = []
    pos_d_f = False
    temp = 'non'
    if opt != 'POS-D':
        for i in range(len(traj_c)):
            #submit prefix
            h0, _ = submit(m0, traj_c[i:i+1], h0)
            presim = np.linalg.norm(Q - h0[-1,0,:])
            sufsim = heuristic_suffix_opt(invQ, Q, traj_c, i+1, opt, each_step_b_c)
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
                h0 = None
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
            h0, _ = submit(m0, traj_c[i:i+1], h0)
            presim = np.linalg.norm(Q - h0[-1,0,:])
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
                h0 = None
                delay = 0
                pos_d_f = False
                pos_d_coll = []
        if subsim == 999999: #for extreme cases
            if pos_d_coll == []:
                h0, _ = submit(m0, traj_c[i:i+1], h0)
                presim = np.linalg.norm(Q - h0[-1,0,:])
                pos_d_coll.append((presim, i))
            sort = sorted(pos_d_coll, key=lambda d: d[0])
            temp = sort[0][1] + 1
            subsim = sort[0][0]
            subtraj = [split_point, (temp-1)]
            
    return subsim, subtraj

def adjust(ap_subsim, ap_subtraj, traj_c, traj_q, opt):
    if opt == 'PSS':
        if ap_subtraj[1] == len(traj_c) - 1:
            _, each_step_f_q = submit(m0, traj_q)
            Q = each_step_f_q[0,-1,:]
            suffix_h, _ = submit(m0, traj_c[ap_subtraj[0]:])
            return np.linalg.norm(Q - suffix_h[-1,0,:]), ap_subtraj
    return ap_subsim, ap_subtraj

if __name__ == '__main__':
    traj_tokens = data_loading('./data/porto_trj.t')
    (cand, query) = pop_random(len(traj_tokens))
    subsim, subtraj, subset = ExactS(traj_tokens[cand], traj_tokens[query])

    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim)
    
    '''
    POS: maintain prefix only O(n)
    POS-D: maintain prefix and delay k steps only O(n)
    PSS: backward-train-model for suffix traj O(n)
    '''
    opt = 'POS-D' #POS, POS-D, PSS
    ap_subsim, ap_subtraj = heuristic(traj_tokens[cand], traj_tokens[query], opt,  delay_K=1)
    ap_subsim, ap_subtraj = adjust(ap_subsim, ap_subtraj, traj_tokens[cand], traj_tokens[query], opt)
    print('ap-sub-trajectory', ap_subtraj)
    print('ap-sub-similarity', ap_subsim)
    print('approximate ratio', ap_subsim/subsim)