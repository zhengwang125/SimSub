from distance import data_loading, pop_random, submit
import numpy as np
import evaluate
from t2vec import args
import random

#random.seed(0)

args.checkpoint =  "./data/best_model_porto.pt"#"best_model_harbin.pt"
args.vocab_size = 18867
m0 = evaluate.model_init(args)

def ExactS(traj_c, traj_q):
    _, each_step_f_q = submit(m0, traj_q)
    Q = each_step_f_q[0,-1,:]
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    subset = {}
    for i in range(len(traj_c)):
         _, each_step_f_c = submit(m0, traj_c[i:])
         for j in range(each_step_f_c.size(1)):
             #print('sub-range:', [i, i+j])
             temp = np.linalg.norm(Q - each_step_f_c[0,j,:])
             subset[(i, i+j)] = temp
             if temp < subsim:
                 subsim = temp
                 subtraj = [i, i+j]
    return subsim, subtraj, subset
    

if __name__ == '__main__':
    traj_tokens = data_loading('./data/porto_trj.t')
    (cand, query) = pop_random(len(traj_tokens))
    subsim, subtraj, subset = ExactS(traj_tokens[cand], traj_tokens[query])
    #print('query:', traj_tokens[query])
    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim)
