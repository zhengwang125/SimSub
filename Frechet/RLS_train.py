from RLS_env import Subtraj
from rl_nn import DeepQNetwork
import numpy as np
import tensorflow as tf
from time import time
from ExactS import ExactS
import pickle

np.random.seed(1)
tf.set_random_seed(1)

path = './data/'

def run_evaluate(elist):
    eva = []
    for e in elist:
        observation, steps = env.reset(e)
        for index in range(1, steps):
            action = RL.online_act(observation)
            observation_, _ = env.step(e, action, index)
            observation = observation_
        res = env.output(index, e)
        if SUBSIM[e] != 0:
            eva.append(res[0]/SUBSIM[e])
    aver_cr = sum(eva)/len(eva)
    print('average competive ratio:', aver_cr)
    return aver_cr
        
def run_subt():
    batch_size = 32
    check = 999999
    REWARD_CL = []
    TR_CR = []
    start = time()
    for episode in range(24200):

        observation, steps = env.reset(episode)

        REWARD = 0.0
        for index in range(1, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            # RL choose action based on observation
            action = RL.act(observation)

            # RL take action and get next observation and reward
            observation_, reward = env.step(episode, action, index)
            
            if reward != 0:                
                REWARD = REWARD + reward

            RL.remember(observation, action, reward, observation_, done)

            if done:
                RL.update_target_model()
#                print("episode: {}/{}, total score: {}"
#                      .format(episode, 600, REWARD))
                break
            if len(RL.memory) > batch_size:
                RL.replay(batch_size)

            # swap observation
            observation = observation_
        
        #check in real time
        #env.output(index, episode)
            
        REWARD_CL.append(REWARD)
        
        # check states
        if SUBSIM[episode] != 0:
            TR_CR.append(env.output(index, episode)[0]/SUBSIM[episode])
            #print('episode', episode, 'ratio', env.output(index, episode)[0],SUBSIM[episode])
        
        if episode % 100 == 0:
             print(episode,'/ 24500', time()-start, 'seconds') 
             aver_cr = run_evaluate([i for i in range(24200, 24500)])
             if aver_cr < 1:
                 continue
             print('Training CR: {}, Validation CR: {}'.format(sum(TR_CR[-100:])/len(TR_CR[-100:]), aver_cr))
             if aver_cr < check or episode % 500 == 0:
                 RL.save('./save/sub-RL-' + str(aver_cr) + '.h5')
                 print('Save model at episode {} with competive ratio {}'.format(episode, aver_cr))
             if aver_cr < check:
                 check = aver_cr
                 print('maintain the current best', check)
                
if __name__ == "__main__":
    # building subtrajectory env
    env = Subtraj(path + 'cand_train', path + 'query_train')
    RL = DeepQNetwork(env.n_features, env.n_actions)
    #RL.load("./save/your_rls_model.h5")
    #Be careful, do not overlap the SUBSIM if you have generated (by dump), too time-consuming for groundtruth
    SUBSIM = []
    for i in range(0, 24500):#
        if i % 1000 == 0:
            print('process', i)
        __subsim, __subtraj, __subset = ExactS(env.cand_train_data[i], env.query_train_data[i])
        SUBSIM.append(__subsim)
    pickle.dump(SUBSIM, open(path + 'SUBSIM', 'wb'), protocol=2)
    SUBSIM = pickle.load(open(path + 'SUBSIM', 'rb'), encoding='bytes')
    run_subt()