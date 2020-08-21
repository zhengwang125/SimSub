import numpy as np
import pickle
from distance import Distance

class Subtraj():
    def __init__(self, cand_train, query_train):
        self.action_space = ['0', '1', '2', '3', '4']
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
        
    def reset(self, episode, label='E'):
        # prefix_state --> [split_point, index]
        # suffix_state --> [index + 1, len - 1]
        # return observation
        self.split_point = 0
        self.DIS = Distance(len(self.cand_train_data[episode]), len(self.query_train_data[episode]))
        self.DIS_R = Distance(len(self.cand_train_data[episode]), len(self.query_train_data[episode]))
        self.length = len(self.cand_train_data[episode])
        self.skip = []
        self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:1], self.query_train_data[episode])
        whole = self.DIS_R.DTW(self.cand_train_data[episode][::-1],self.query_train_data[episode][::-1]) 
        self.sufsim = self.DIS_R.D[self.length-2,-1]#self.DIS_R.DTW(self.cand_train_data[episode][1:][::-1],self.query_train_data[episode][::-1])
        #print('reset',self.sufsim,self.DIS_R.D[self.length-2,-1])
        #self.DIS.DTW(self.cand_train_data[episode], self.query_train_data[episode])
        observation = np.array([whole, self.presim, self.sufsim]).reshape(1,-1)
        
        self.subsim = min(whole, self.presim, self.sufsim)
        
        #print('episode', episode, whole, self.presim, self.sufsim)
        
        if self.subsim == whole:
            self.subtraj = [0, self.length - 1]
            
        if self.subsim == self.presim:
            self.subtraj = [0, 0]
        
        if self.subsim == self.sufsim:
            self.subtraj = [1, self.length - 1]
            
        if label == 'T':
            self.REWARD_DIS = Distance(len(self.cand_train_data[episode]), len(self.query_train_data[episode]))
            self.presim_real = self.REWARD_DIS.DTW(self.cand_train_data[episode][self.split_point:1], self.query_train_data[episode])
            self.subsim_real = self.subsim
        return observation, self.length, -1
        
    def step(self, episode, action, index, label='E'):
        if action == 0: #non-split 
            #state transfer
            if index == self.length - 1:
                done = True
            else:
                done = False 
            self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode], self.skip)
            self.sufsim = self.DIS_R.D[self.length-2-index,-1]#self.DIS_R.DTW(self.cand_train_data[episode][(index+1):][::-1],self.query_train_data[episode][::-1])
            #print('A0', self.sufsim, self.DIS_R.D[self.length-2-index,-1])
            if (index+1) == self.length:
                self.sufsim = self.presim
            observation = np.array([self.subsim, self.presim, self.sufsim]).reshape(1,-1)
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if self.sufsim < self.subsim:
                self.subsim = self.sufsim
                self.subtraj = [index + 1, self.length - 1]
            
            if label == 'T':
                last_subsim = self.subsim_real
                self.presim_real = self.REWARD_DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode])
                self.subsim_real = min(self.presim_real, self.sufsim, last_subsim)
                self.RW = last_subsim - self.subsim_real            
                #print('action0', self.RW)
                #print(self.presim, self.presim_real)
            
            return observation, self.RW, done, -1
        
        if action == 1: #split
            if index == self.length - 1:
                done = True
            else:
                done = False 
            self.skip = []
            self.split_point = index
            self.DIS = Distance(len(self.cand_train_data[episode][self.split_point:]), len(self.query_train_data[episode]))
            #state transfer
            self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode], self.skip)
            self.sufsim = self.DIS_R.D[self.length-2-index,-1]#self.DIS_R.DTW(self.cand_train_data[episode][(index+1):][::-1],self.query_train_data[episode][::-1])            
            #print('A1', self.sufsim, self.DIS_R.D[self.length-2-index,-1])
            if (index+1) == self.length:
                self.sufsim = self.presim
            observation = np.array([self.subsim, self.presim, self.sufsim]).reshape(1,-1)
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if self.sufsim < self.subsim:
                self.subsim = self.sufsim
                self.subtraj = [index + 1, self.length - 1]
            
            if label == 'T':
                last_subsim = self.subsim_real
                self.REWARD_DIS = Distance(len(self.cand_train_data[episode][self.split_point:]), len(self.query_train_data[episode]))
                self.presim_real = self.REWARD_DIS.DTW(self.cand_train_data[episode][self.split_point:(index+1)], self.query_train_data[episode])
                self.subsim_real = min(self.presim_real, self.sufsim, last_subsim)
                self.RW = last_subsim - self.subsim_real         
                #print('action1', self.RW)
                #print(self.presim, self.presim_real)
                
            return observation, self.RW, done, -1
        
        if action > 1: #skip
            INX = min(index + action - 1, self.length - 1)
            if INX == self.length - 1:
                done = True
            else:
                done = False
            for i in range(index,INX):
                self.skip.append(list(self.cand_train_data[episode][i]))
            self.presim = self.DIS.DTW(self.cand_train_data[episode][self.split_point:(INX+1)], self.query_train_data[episode], self.skip)
            self.sufsim = self.DIS_R.D[self.length-2-INX,-1] #self.DIS_R.DTW(self.cand_train_data[episode][(INX+1):][::-1],self.query_train_data[episode][::-1])
            #print('A2', self.sufsim, self.DIS_R.D[self.length-2-INX,-1])
            if (INX+1) == self.length:
                self.sufsim = self.presim
            observation = np.array([self.subsim, self.presim, self.sufsim]).reshape(1,-1)
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, INX]
            if self.sufsim < self.subsim:
                self.subsim = self.sufsim
                self.subtraj = [INX + 1, self.length - 1]
            
            if label == 'T':
                last_subsim = self.subsim_real
                self.presim_real = self.REWARD_DIS.DTW(self.cand_train_data[episode][self.split_point:(INX+1)], self.query_train_data[episode])
                self.subsim_real = min(self.presim_real, self.sufsim, last_subsim)
                self.RW = last_subsim - self.subsim_real    
                #print(self.presim, self.presim_real)
            return observation, self.RW, done, INX
            
    def output(self, index, episode, label='E'):
        #print('check', self.subsim, self.subtraj)
        if label == 'T':
            #print('check', self.subsim, self.subtraj, self.subsim_real)
            return [self.subsim_real, self.subtraj]
        if label == 'E':
            self.DIS = Distance(len(self.cand_train_data[episode][self.subtraj[0]:self.subtraj[1]+1]), len(self.query_train_data[episode]))
            self.subsim_real = self.DIS.DTW(self.cand_train_data[episode][self.subtraj[0]:self.subtraj[1]+1], self.query_train_data[episode])
            return [self.subsim_real, self.subtraj]