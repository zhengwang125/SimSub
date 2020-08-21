import h5py
import random
import pickle
from sklearn.model_selection import train_test_split
from distance import pop_random

random.seed(0)

if __name__ == '__main__':
    f = h5py.File('./data/porto_querydb.h5','r')
    Cand = []
    Query = []
    for i in range(35000):
        (cand, query) = pop_random(f['/db/num/'].value)
        traj_C = f['/db/trips/'+str(cand)].value
        traj_Q = f['/db/trips/'+str(query)].value
        Cand.append(traj_C)
        Query.append(traj_Q)
    cand_train, cand_test, query_train, query_test = train_test_split(Cand, Query, random_state=1, train_size=0.7)    
    pickle.dump(cand_train, open('./data/cand_train', 'wb'), protocol=2)
    pickle.dump(cand_test, open('./data/cand_test', 'wb'), protocol=2)
    pickle.dump(query_train, open('./data/query_train', 'wb'), protocol=2)
    pickle.dump(query_test, open('./data/query_test', 'wb'), protocol=2)