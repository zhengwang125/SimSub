import numpy as np
import tensorflow as tf
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Deep Q Network off-policy

class DeepQNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(25, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate)) # self._huber_loss
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            #print('->act by random')
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        #print('->act by model')
        #print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])  # returns action
    
    def relu(self, x):
        return np.maximum(0,x)
    
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def fast_online_act(self, state):
        o1 = self.relu(np.dot(state, self.w1) + self.b1)
        o2 = self.sigmoid(np.dot(o1, self.w2) + self.b2)
        #print(o2)
        return np.argmax(o2[0])
    
    def online_act(self, state):
        act_values = self.model.predict(state)
        #print('->online act by model')
        #print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            #print('replay', state, action, reward, state)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                #a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                #target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0, shuffle=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.w1 = self.model.get_weights()[0]
        self.b1 = self.model.get_weights()[1]
        self.w2 = self.model.get_weights()[2]
        self.b2 = self.model.get_weights()[3]

    def save(self, name):
        self.model.save_weights(name)
    
    def plot_score(self, cl):
        plt.plot(np.arange(len(cl)), cl)
        plt.ylabel('Score')
        plt.xlabel('Training episodes')
        plt.show()
    
    def plot_cr(self, cl):
        plt.plot(np.arange(len(cl)), cl)
        plt.ylabel('CR')
        plt.xlabel('Training episodes')
        plt.show()