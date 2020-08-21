import torch
import torch.nn as nn
from torch.autograd import Variable
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import constants
from time import time
import numpy as np

def evaluate(src, model, max_length):
    """
    evaluate one source sequence
    """
    m0, m1 = model
    length = len(src)
    src = Variable(torch.LongTensor(src))
    ## (seq_len, batch)
    src = src.view(-1, 1)
    length = Variable(torch.LongTensor([[length]]))

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)
    ## running the decoder step by step with BOS as input
    input = Variable(torch.LongTensor([[constants.BOS]]))
    trg = []
    for _ in range(max_length):
        ## `h` is updated for next iteration
        o, h = m0.decoder(input, h, H)
        o = o.view(-1, o.size(2)) ## => (1, hidden_size)
        o = m1(o) ## => (1, vocab_size)
        ## the most likely word
        _, word_id = o.data.topk(1)
        word_id = word_id[0][0]
        if word_id == constants.EOS:
            break
        trg.append(word_id)
        ## update `input` for next iteration
        input = Variable(torch.LongTensor([[word_id]]))
    return trg

#checkpoint = torch.load("checkpoint.pt")
#m0.load_state_dict(checkpoint["m0"])
#m1.load_state_dict(checkpoint["m1"])
#
#src = [9, 11, 14]
#trg = evaluate(src, (m0, m1), 20)
#trg

def evaluator(args):
    """
    do evaluation interactively
    """
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        while True:
            try:
                print("> ", end="")
                src = input()
                src = [int(x) for x in src.split()]
                trg = evaluate(src, (m0, m1), args.max_length)
                print(" ".join(map(str, trg)))
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

def model_init(args):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        #print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            print('mo by cuda')
            m0.cuda()
        m0.eval()
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return m0

def t2vec(m0, args, h0 = None, trajdata = None):
    vecs = []
    ovecs = []
    scaner = DataOrderScaner(os.path.join(args.data, "trj.t"), args.t2vec_batch)
    scaner.load(max_num_line = 0, trajdata = trajdata)
#    i = 0
    while True:
#        if i % 10 == 0:
#           print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
#        i = i + 1
        src, lengths, invp = scaner.getbatch()
#        print('src', src)
#        print('length',lengths)
#        print('invp', invp)
        if src is None: break
        src, lengths = Variable(src), Variable(lengths)
        if torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        if (not h0 is None) and (torch.cuda.is_available()):
            h0 = h0.cuda()
        h, o = m0.encoder(src, lengths, h0)
#        print(h[-1][0])
#        print(h[-1][1])
#        print(o[-1][0])
#        print(o[-1][1])
        ## (num_layers, batch, hidden_size * num_directions)
        #h = m0.encoder_hn2decoder_h0(h)
        ## (batch, num_layers, hidden_size * num_directions)
        h = h.transpose(0, 1).contiguous()
        o = o.transpose(0, 1).contiguous()
#        print(h.size())
#        print(o.size())
        ## (batch, *)
        #h = h.view(h.size(0), -1)
        vecs.append(h[invp].cpu().data)
        ovecs.append(o[invp].cpu().data)
    ## (num_seqs, num_layers, hidden_size * num_directions)
    vecs = torch.cat(vecs)
    ovecs = torch.cat(ovecs)
    ## (num_layers, num_seqs, hidden_size * num_directions)
    vecs = vecs.transpose(0, 1).contiguous()
#    path = os.path.join(args.data, str(args.seq_reverse)+"trj.h5")
#    print("=> saving vectors into {}".format(path))
#    with h5py.File(path, "w") as f:
#        for i in range(m0.num_layers):
#            f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()
#    torch.save(vecs.data, path)
    return vecs.data, ovecs.data

#args = FakeArgs()
#args.t2vec_batch = 128
#args.num_layers = 2
#args.hidden_size = 64
#vecs = t2vec(args)
#vecs
'''
def t2vec(m0, args, h0 = None, trajdata = None):
    src = torch.LongTensor(np.array(trajdata[0], dtype=np.int32)).view(-1, 1)
    lengths = torch.LongTensor([len(trajdata[0])]).view(-1, 1)
    vecs = []
    ovecs = []
#    print(src.size())
#    print(lengths.size())
    src, lengths = Variable(src), Variable(lengths)
        
    if torch.cuda.is_available():
        src, lengths = src.cuda(), lengths.cuda()
    
    if (not h0 is None) and (torch.cuda.is_available()):
        h0 = h0.cuda()
    
    h, o = m0.encoder(src, lengths, h0)
    ## (num_layers, batch, hidden_size * num_directions)
    #h = m0.encoder_hn2decoder_h0(h)
    ## (batch, num_layers, hidden_size * num_directions)
    h = h.transpose(0, 1).contiguous()
    o = o.transpose(0, 1).contiguous()
    vecs.append(h.cpu().data)
    ovecs.append(o.cpu().data)
    vecs = torch.cat(vecs)
    ovecs = torch.cat(ovecs)
    vecs = vecs.transpose(0, 1).contiguous()
    return vecs.data, ovecs.data
'''