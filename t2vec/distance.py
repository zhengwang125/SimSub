from t2vec import args
import evaluate
import random
import torch
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from time import time

def data_loading(path):
    traj_tokens = []
    for line in open(path):
        traj_tokens.append(line.strip('\n').split(' '))
    return traj_tokens

def pop_random(lst):
    idx1 = random.randrange(0, lst)
    idx2 = random.randrange(0, lst)
    return (idx1,idx2)

#def submit(m0, traj, h0=None):
#    if type(traj[0]) != list:
#        traj = [traj]
##    if args.io_by_disk:
##        f = open(args.data + '\\trj.t', 'w')
##        for item in traj:
##            for point in item:
##                f.write(point + ' ')
##            f.write('\n')
##        f.close()
##        res_f, each_step_f = evaluate.t2vec(m0, args, h0)
##    else:
#    res_f, each_step_f = evaluate.t2vec(m0, args, h0, traj)
#    print('res_f', res_f.size())
#    print('each_step_f', each_step_f.size())
#    return res_f, each_step_f

def submit(m0, traj, h0=None): #a quick version of t2vec
    srcdata = [int(item) for item in traj]
    srcdata = np.array(srcdata)
    srcdata = [srcdata]
    #embed = m0.embedding(torch.LongTensor(srcdata).cuda().reshape(-1,1))
    embed = m0.embedding(torch.LongTensor(srcdata).reshape(-1,1))
    if (not h0 is None) and (torch.cuda.is_available()):
        h0 = h0.cuda()
        #print('h0 by cuda')
    if (not embed is None) and (torch.cuda.is_available()):
        embed = embed.cuda()
        #print('embed by cuda')
    output, hn = m0.encoder.rnn(embed, h0)
    output = output.transpose(0, 1).contiguous()
#    print('hn', hn.size())
#    print('output', output.size())
    return hn.cpu().data, output.cpu().data

def generate_suffix(text, sign=''):
    suffixs = []
    for _i in range(len(text)):
        if _i == 0:
            suffixs.append(text)
        else:
            suffixs.append(text[_i:])
    if not sign:
        return suffixs
    else:
        return suffixs+[sign]
    