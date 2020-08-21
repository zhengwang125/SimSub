import numpy as np
import pickle
from distance import Distance

class Subtraj():
    def __init__(self, cand_train, query_train):
        self.action_space = ['0', '1']
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.cand_train_name = cand_train
        self.query_train_name = query_train
        self.presim = 0
        self.sufsim = 0
        self.RW = 0.0
        self._load()

    def _load(self):
        self.cand_train_data = pickle.load(open(self.cand_train_name, 'rb'), encoding='bytes')
        self.query_train_data = pickle.load(open(self.query_train_name, 'rb'), encoding='bytes')
        
    def reset(self, episode):
        # prefix_state --> [split_point, index]
        # suffix_state --> [index + 1, len - 1]
        # return observation
        self.split_point = 0
        self.DIS = Distance(len(self.cand_train_data[episode]), len(self.query_train_data[episode]))
        self.DIS_R = Distance(len(self.cand_train_data[episode]), len(self.query_train_data[episode]))
        self.length = len(self.cand_train_data[episode])
        
        self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:1], self.query_train_data[episode])
        self.sufsim = self.DIS_R.DTW(self.cand_train_data[episode][1:][::-1],self.query_train_data[episode][::-1])
        whole = self.DIS_R.DTW(self.cand_train_data[episode][::-1],self.query_train_data[episode][::-1]) #self.DIS.DTW(self.cand_train_data[episode], self.query_train_data[episode])
        observation = np.array([whole, self.presim, self.sufsim]).reshape(1,-1)
        
        self.subsim = min(whole, self.presim, self.sufsim)
        #print('episode', episode, whole, self.presim, self.sufsim)
        
        if self.subsim == whole:
            self.subtraj = [0, self.length - 1]
            
        if self.subsim == self.presim:
            self.subtraj = [0, 0]
        
        if self.subsim == self.sufsim:
            self.subtraj = [1, self.length - 1]
        
        return observation, self.length
        
    def step(self, episode, action, index):
        if action == 0: #non-split 
            #state transfer
            self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode])
            self.sufsim = self.DIS_R.DTW(self.cand_train_data[episode][(index+1):][::-1],self.query_train_data[episode][::-1])
            if (index+1) == self.length:
                self.sufsim = self.presim
            observation = np.array([self.subsim, self.presim, self.sufsim]).reshape(1,-1)
            
            last_subsim = self.subsim
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if self.sufsim < self.subsim:
                self.subsim = self.sufsim
                self.subtraj = [index + 1, self.length - 1]
                
            self.RW = last_subsim - self.subsim            
            #print('action0', self.RW)
            return observation, self.RW
        
        if action == 1: #split
            self.split_point = index
            self.DIS = Distance(len(self.cand_train_data[episode][self.split_point:]), len(self.query_train_data[episode]))
            #state transfer
            self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode])
            self.sufsim = self.DIS_R.DTW(self.cand_train_data[episode][(index+1):][::-1],self.query_train_data[episode][::-1])            
            if (index+1) == self.length:
                self.sufsim = self.presim
            observation = np.array([self.subsim, self.presim, self.sufsim]).reshape(1,-1)
            
            last_subsim = self.subsim
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if self.sufsim < self.subsim:
                self.subsim = self.sufsim
                self.subtraj = [index + 1, self.length - 1]
            
            self.RW = last_subsim - self.subsim         
            #print('action1', self.RW)
            return observation, self.RW

    def output(self, index, episode):
        #print('check', self.subsim, self.subtraj)
        return [self.subsim, self.subtraj]