from distance import data_loading, pop_random, submit
import numpy as np
import evaluate
import random
from t2vec import args
from ExactS import ExactS

#random.seed(0)

args.checkpoint = "./data/best_model_porto.pt"
args.vocab_size = 18867
m0 = evaluate.model_init(args)

def SizeS(traj_c, traj_q, par=5):
    L = len(traj_q)
    L_lo = min(len(traj_c), int((L - par)))
    L_up = min(len(traj_c), int((L + par)))
    #print('Lower upper', [L_lo, L_up])
    _, each_step_f_q = submit(m0, traj_q)
    Q = each_step_f_q[0,-1,:]
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    for i in range(len(traj_c)):
         _, each_step_f_c = submit(m0, traj_c[i:i+L_up])
         for j in range(each_step_f_c.size(1)):
             #print('sub-range:', [i, i+j])
             if (j + 1) < L_lo:
                 continue
             temp = np.linalg.norm(Q - each_step_f_c[0,j,:])
             if temp < subsim:
                 subsim = temp
                 subtraj = [i, i+j]
    return subsim, subtraj
    
if __name__ == '__main__':
    traj_tokens = data_loading('./data/porto_trj.t')
    (cand, query) = pop_random(len(traj_tokens))
    par = 1
    ap_subsim, ap_subtraj = SizeS(traj_tokens[cand], traj_tokens[query], par)
    subsim, subtraj, subset = ExactS(traj_tokens[cand], traj_tokens[query])
    #print('query:', traj_tokens[query])
    print('sub-trajectory', ap_subtraj)
    print('sub-similarity', ap_subsim)
    #print('candidate', traj_tokens[cand][ap_subtraj[0]:ap_subtraj[1] + 1])
    print('approximate ratio', ap_subsim/subsim)