import numpy as np
import pickle
import evaluate
from t2vec import args
from distance import submit, generate_suffix

args.checkpoint = "./data/best_model_porto.pt"
args.vocab_size = 18867
m0 = evaluate.model_init(args)

class Subtraj():
    def __init__(self, cand_train, query_train):
        self.action_space = ['0', '1']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.cand_train_name = cand_train
        self.query_train_name = query_train
#        self.query_state_name = query_state
#        self.query_state_back_name = query_state_back
#        self.cand_state_back_name = cand_state_back
#        self.cand_state_forw_name = cand_state_forw
        self.presim = 0
        self.sufsim = 0
        self.RW = 0.0
        self.delay = 0
        self._load()

    def _load(self):
        self.cand_train_data = pickle.load(open(self.cand_train_name, 'rb'), encoding='bytes')
        self.query_train_data = pickle.load(open(self.query_train_name, 'rb'), encoding='bytes')
#        self.query_state_data = pickle.load(open(self.query_state_name, 'rb'), encoding='bytes')
#        self.query_state_back_data = pickle.load(open(self.query_state_back_name, 'rb'), encoding='bytes')
#        self.cand_state_back_data = pickle.load(open(self.cand_state_back_name, 'rb'), encoding='bytes')
#        self.cand_state_forw_data = pickle.load(open(self.cand_state_forw_name, 'rb'), encoding='bytes')
        
    def reset(self, episode, label='E'):
        # prefix_state --> [split_point, index]
        # suffix_state --> [index + 1, len - 1]
        # return observation
        self.query_state_data, _ = submit(m0, self.query_train_data[episode])
        #_, self.query_state_back_data = submit(m0, self.query_train_data[episode][::-1])
        #_, self.cand_state_back_data = submit(m0, self.cand_train_data[episode][::-1])
        
        self.split_point = 0
        self.h0 = None
        self.h0, _ = submit(m0, self.cand_train_data[episode][0:1], self.h0)
        self.length = len(self.cand_train_data[episode])
        
        #whole = np.linalg.norm(self.query_state_back_data[0, -1] - self.cand_state_back_data[0, -1])
        self.presim = np.linalg.norm(self.query_state_data[-1] - self.h0[-1])
        #self.sufsim = np.linalg.norm(self.query_state_back_data[0, -1] - self.cand_state_back_data[0, self.length - 2])
        observation = np.array([self.presim, self.presim]).reshape(1,-1)
        
        self.subsim = min(self.presim, self.presim)
        #print('episode', episode, whole, self.presim, self.sufsim)
        
#        if self.subsim == whole:
#            self.subtraj = [0, self.length - 1]
            
        if self.subsim == self.presim:
            self.subtraj = [0, 0]
        
#        if self.subsim == self.sufsim:
#            self.subtraj = [1, self.length - 1]
        
        if label == 'T':
            #t = generate_suffix(self.cand_train_data[episode])
            #self.cand_state_forw_data, _ = submit(m0, t)
            #whole_real = np.linalg.norm(self.query_state_data[-1] - self.cand_state_forw_data[-1, 0])
            #suffix_real = np.linalg.norm(self.query_state_data[-1] - self.cand_state_forw_data[-1, 1])
            self.subsim_real = min(self.presim, self.presim)#, suffix_real
        
        return observation, self.length
        #return np.concatenate((self.query_state_data[-1].numpy().reshape(1, -1), self.h0[-1].numpy().reshape(1, -1), self.cand_state_back_data[0, self.length - 2].numpy().reshape(1, -1)), 1), self.length
        
        
    def step(self, episode, action, index, label='E'):
        if action == 0: #non-split 
            #state transfer
            self.h0, _ = submit(m0, self.cand_train_data[episode][index:index + 1], self.h0)
            self.presim = np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
#            self.sufsim = np.linalg.norm(self.query_state_back_data[0, -1] - self.cand_state_back_data[0, self.length - index - 2])
            #print(self.sufsim)
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)#
            
#            if self.sufsim < self.subsim:
#                self.subsim = self.sufsim
#                self.subtraj = [index + 1, self.length - 1]
                
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
            
            if label == 'T':
                last_subsim = self.subsim_real
                #if self.subtraj[1] == self.length - 1 and self.subtraj[0] < self.length: #reward may record error
                self.subsim_real = min(last_subsim, self.presim)#, np.linalg.norm(self.query_state_data[-1] - self.cand_state_forw_data[-1, self.subtraj[0]])
#                else:
#                    self.subsim_real = self.subsim
                    #print('0 Rw change')
                self.RW = last_subsim - self.subsim_real              
                    
            return observation, self.RW
            #return np.concatenate((self.query_state_data[-1].numpy().reshape(1, -1), self.h0[-1].numpy().reshape(1, -1), self.cand_state_back_data[0, self.length - index - 2].numpy().reshape(1, -1)), 1), self.RW
        if action == 1: #split
            self.split_point = index
            self.h0 = None
            self.h0, _ = submit(m0, self.cand_train_data[episode][index:index + 1], self.h0)
            
            #state transfer
            self.presim = np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
#            self.sufsim = np.linalg.norm(self.query_state_back_data[0, -1] - self.cand_state_back_data[0, self.length - index - 2])            
            #print(self.sufsim)
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
            
#            if self.sufsim < self.subsim:
#                self.subsim = self.sufsim
#                self.subtraj = [index + 1, self.length - 1]
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if label == 'T':
                last_subsim = self.subsim_real
                #if self.subtraj[1] == self.length - 1 and self.subtraj[0] < self.length: #reward may record error
                self.subsim_real = min(last_subsim, self.presim)#, np.linalg.norm(self.query_state_data[-1] - self.cand_state_forw_data[-1, self.subtraj[0]])
                    #print('1 Rw change')
#                else:
#                    self.subsim_real = self.subsim
                self.RW = last_subsim - self.subsim_real
            
            return observation, self.RW
            #return np.concatenate((self.query_state_data[-1].numpy().reshape(1, -1), self.h0[-1].numpy().reshape(1, -1), self.cand_state_back_data[0, self.length - index - 2].numpy().reshape(1, -1)), 1), self.RW

    def output(self, index, episode, label='E'):
#        if self.subtraj[1] == self.length - 1 and index == self.length - 1:
#            suffix_h, _ = submit(m0, self.cand_train_data[episode][self.subtraj[0]:])
#            self.subsim = np.linalg.norm(self.query_state_data[-1] - suffix_h[-1,0,:])
        if label == 'T':
            print('check', self.subsim, self.subtraj, self.subsim_real)
        return [self.subsim, self.subtraj]