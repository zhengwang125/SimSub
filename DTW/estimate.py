from Splitting_based import heuristic
from ExactS import ExactS
import pickle
import RLS_train as RLS
import RLS_Skip_train as RLS_Skip
from SizeS import SizeS

path = './data/'

def run_effective_rls_skip(elist):
    eva = []
    rank = []
    relative_rank = []
    for e in elist:
        #print('e in elist', e)
        observation, steps, INX  = env.reset(e)
        total = (steps) * (steps + 1) / 2
        for index in range(1, steps):
            if index <= INX:
                #print('skip')
                continue
            action = RL.fast_online_act(observation)
            observation_, _, done, INX = env.step(e, action, index)
            observation = observation_
        res = env.output(index, e)   
        if SUBSIM_TEST[e] != 0:
            eva.append(res[0]/SUBSIM_TEST[e])
        elif res[0] == SUBSIM_TEST[e]:
            eva.append(1.0)
        if tuple(res[1]) in SUBSIM_RANK[e]:
            t = SUBSIM_RANK[e].index(tuple(res[1])) + 1
            #print(t)
            rank.append(t)
            relative_rank.append(t*1.0/total)
        else:
            print((tuple(res[1]), res[0]))
        if e + 1 == 10000:
            print('average competive ratio:', sum(eva)/len(eva))
            print('mean rank:', sum(rank)/len(rank))
            print('aver_relative_rank:', sum(relative_rank)/len(relative_rank))
            break

def run_effective_rls(elist):
    eva = []
    rank = []
    relative_rank = []
    for e in elist:
        observation, steps = env.reset(e)  
        total = (steps) * (steps + 1) / 2
        for index in range(1, steps):            
            action = RL.fast_online_act(observation)
            observation_, _ = env.step(e, action, index)
            observation = observation_            
        res = env.output(index, e)   
        if SUBSIM_TEST[e] != 0:
            eva.append(res[0]/SUBSIM_TEST[e])
        elif res[0] == SUBSIM_TEST[e]:
            eva.append(1.0)
        if tuple(res[1]) in SUBSIM_RANK[e]:
            t = SUBSIM_RANK[e].index(tuple(res[1])) + 1
            #print(t)
            rank.append(t)
            relative_rank.append(t*1.0/total)
        else:
            print((tuple(res[1]), res[0]))
        if e + 1 == 10000:
            print('average competive ratio:', sum(eva)/len(eva))
            print('mean rank:', sum(rank)/len(rank))
            print('aver_relative_rank:', sum(relative_rank)/len(relative_rank))
            break

if __name__ == '__main__':
    cand_test_data = pickle.load(open(path + 'cand_test', 'rb'), encoding='bytes')
    query_test_data = pickle.load(open(path + 'query_test', 'rb'), encoding='bytes')
    length = len(cand_test_data)
    elist = [i for i in range(10000)]

    '''
    POS: maintain prefix only O(n)
    POS-D: maintain prefix and delay k steps only O(n)
    PSS: backward-train-model for suffix traj O(n)
    '''
    
    SUBSIM_TEST = []
    SUBSIM_RANK = []
    for i in elist:
        if i % 1000 == 0:
            print('process', i)
        subsim, subtraj, subset = ExactS(cand_test_data[i], query_test_data[i])
        #print('lens cand:', len(cand_test_data[i]), 'lens sub:', len(subset))
        SUBSIM_TEST.append(subsim)
        subsort = sorted(subset.items(), key=lambda d: d[1])
        SUBSIM_RANK.append([j[0] for j in subsort])
    
    pickle.dump(SUBSIM_TEST, open(path + 'SUBSIM_TEST', 'wb'), protocol=2)
    pickle.dump(SUBSIM_RANK, open(path + 'SUBSIM_RANK', 'wb'), protocol=2)
    
    SUBSIM_TEST = pickle.load(open(path + 'SUBSIM_TEST', 'rb'), encoding='bytes')
    SUBSIM_RANK = pickle.load(open(path + 'SUBSIM_RANK', 'rb'), encoding='bytes')
    
    for oo in ['SizeS', 'POS-D', 'POS', 'PSS']: 
        print('algorithm:', oo)
        opt = oo
        eva = []
        rank = []
        relative_rank = []
        for i in elist:
            #print('steps:', len(cand_test_data[i]), len(SUBSIM_RANK[i]))
            total = len(cand_test_data[i]) * (len(cand_test_data[i]) + 1) / 2
            if opt == 'SizeS':
                ap_subsim, ap_subtraj = SizeS(cand_test_data[i], query_test_data[i])
            else:
                ap_subsim, ap_subtraj = heuristic(cand_test_data[i], query_test_data[i], opt)
            
            if SUBSIM_TEST[i] != 0:
                eva.append(ap_subsim/SUBSIM_TEST[i])
            elif ap_subsim == SUBSIM_TEST[i]:
                eva.append(1.0)
            if tuple(ap_subtraj) in SUBSIM_RANK[i]:
                t = SUBSIM_RANK[i].index(tuple(ap_subtraj)) + 1
                rank.append(t)
                relative_rank.append(t*1.0/total)
            if i + 1 == 10000:
                print('average competive ratio:', sum(eva)/len(eva))
                print('mean rank:', sum(rank)/len(rank))
                print('relative rank:', sum(relative_rank)/len(relative_rank))
                break
    env = RLS.Subtraj(path+'cand_test', path +'query_test')
    RL = RLS.DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./save/your_rls_model.h5')
    print('algorithm:', 'RLS')
    run_effective_rls(elist)
    
    env = RLS_Skip.Subtraj(path+'cand_test', path +'query_test')
    RL = RLS_Skip.DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./save/your_rls_skip_model.h5')
    print('algorithm:', 'RLS-Skip')
    run_effective_rls_skip(elist)