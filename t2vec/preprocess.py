from distance import data_loading, pop_random
import pickle
from sklearn.model_selection import train_test_split
import random

random.seed(0)

if __name__ == '__main__':
    traj_tokens = data_loading('./data/porto_trj.t')
    Cand = []
    Query = []
    for i in range(35000):
        (cand, query) = pop_random(len(traj_tokens))
        Cand.append((traj_tokens[cand]))
        Query.append((traj_tokens[query]))
    cand_train, cand_test, query_train, query_test = train_test_split(Cand, Query, random_state=1, train_size=0.7)    
    pickle.dump(cand_train, open('./data/cand_train', 'wb'), protocol=2)
    pickle.dump(cand_test, open('./data/cand_test', 'wb'), protocol=2)
    pickle.dump(query_train, open('./data/query_train', 'wb'), protocol=2)
    pickle.dump(query_test, open('./data/query_test', 'wb'), protocol=2)
    