import os, time
import numpy as np
import random
from random import randint


import torch
import torch.nn as nn
from torch.autograd import Variable

GLOVE_PATH = "dataset/GloVe/"


# Create dictionary
def get_dict(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>']   = 1e9 + 4
    words['</s>']  = 1e9 + 3
    words['<p>']   = 1e9 + 2
    #words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1]) # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i
    
    return id2word, word2id

# Get batch
def get_batch(batch_sentences, index_pad = 1e9 + 2):
    # sort sentences by length
    lengths = np.array([sentence.size(0) for sentence in batch_sentences])
    batch   =  torch.LongTensor(lengths.max(), len(batch_sentences)).fill_(int(index_pad))
    for i in xrange(len(batch_sentences)):
        batch[:lengths[i], i] = torch.LongTensor(batch_sentences[i])
    return batch, lengths
    # size : seqlen * bsize
    
    
def get_batch2(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths 


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print 'Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print 'Vocab size : {0}'.format(len(word_vec))
    return word_vec




























# Get lookup-table with GloVe vectors
def get_lut_glove(glove_type, glove_path, word2id):
    word_emb_dim = int(glove_type.split('.')[1].split('d')[0])
    src_embeddings = nn.Embedding(len(word2id), word_emb_dim, padding_idx=word2id['<p>'])
    #src_embeddings.weight.data.fill_(0)

    n_words_with_glove = 0
    last_time = time.time()
    words_found = {}
    words_not_found = []
    
    # initializing lut with GloVe vectors,
    # words that do not have GloVe vectors have random vectors.
    with open(glove_path + 'glove.' + glove_type + '.txt') as f:
        for line in f:
            word = line.split(' ', 1)[0]
            if word in word2id:
                glove_vect = torch.FloatTensor(list(map(float, line.split(' ', 1)[1].split(' '))))
                src_embeddings.weight.data[word2id[word]].copy_(torch.FloatTensor(glove_vect))
                n_words_with_glove += 1
                words_found[word] = ''
    
    # get words with no GloVe vectors.
    for word in word2id:
        if word not in words_found:
            words_not_found.append(word)
            
    print 'GLOVE : Found ' +  str(len(words_found)) + ' words with GloVe vectors, out of ' +\
                                str(len(word2id)) + ' words in vocabulary'
    print 'GLOVE : Took ' + str(round(time.time()-last_time,2)) + ' seconds.'
    rdm_idx = 0 if len(words_not_found)<8 else randint(0, len(words_not_found) - 1 - 7)
    print 'GLOVE : 7 words in word2id without GloVe vectors : ' + str(words_not_found[rdm_idx:rdm_idx + 7])
    return word_emb_dim, src_embeddings.cuda(), words_not_found


def permutation_per_batch(n, batch_size, block_size=8):
    # If dataset is sorted : get permutation to minimize padding
    ## 1) Splits the range(n) in blocks of size block_size*batch_size.
    ## 2) Shuffle the blocks, and returns a sequence of indexes.
    ## 3) Goal : minimize padding
    block_idx = []
    for start_idx in range(0, n, block_size*batch_size): block_idx.append(\
                           range(start_idx, min(start_idx+block_size*batch_size, n)))
    for block in block_idx: random.shuffle(block)
    lastblock = block_idx.pop(-1)
    random.shuffle(lastblock)
    random.shuffle(block_idx)

    permutation = [idx for block in block_idx for idx in block]
    assert len(permutation) % batch_size == 0, len(permutation)
    permutation = permutation + lastblock
    assert len(permutation) == n
    return permutation






def get_nli2(data_path):
    s1 = {}
    s2 = {}
    target = {}
    
    dico_label = {'entailment':0,  'neutral':1, 'contradiction':2}
    
    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path, 'labels.' + data_type)
        
        s1[data_type]['sent'] = [line.rstrip() for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')] for line in open(target[data_type]['path'], 'r')])
        
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == len(target[data_type]['data'])
        
        print '** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                            data_type.upper(), len(s1[data_type]['sent']), data_type)
        
        
    train = {'s1':s1['train']['sent'], 's2':s2['train']['sent'], 'label':target['train']['data']}
    dev = {'s1':s1['dev']['sent'], 's2':s2['dev']['sent'], 'label':target['dev']['data']}
    test  = {'s1':s1['test']['sent'] , 's2':s2['test']['sent'] , 'label':target['test']['data'] }
    return train, dev, test

























def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}
    
    dico_label = {'entailment':0,  'neutral':1, 'contradiction':2}
    
    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path, 'labels.' + data_type)
        
        s1[data_type]['sent'] = [line.rstrip().split() for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip().split() for line in open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = [dico_label[line.rstrip('\n')] for line in open(target[data_type]['path'], 'r')]
        
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == len(target[data_type]['data'])
        
        # sort to minimize padding for "block" sampling
        sorted_by_s2 = sorted(zip(s2[data_type]['sent'], s1[data_type]['sent'],target[data_type]['data']),\
                                key=lambda z:(len(z[0]), len(z[1]), z[2]))
        s2[data_type]['sent'] = [x for (x,y,z) in sorted_by_s2]
        s1[data_type]['sent'] = [y for (x,y,z) in sorted_by_s2]
        target[data_type]['data'] = np.array([z for (x,y,z) in sorted_by_s2])
        
        
        print '** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                            data_type.upper(), len(s1[data_type]['sent']), data_type)
        
    id2word, word2id = get_dict(s1['train']['sent'] + s2['train']['sent'] +
                                         s1['dev']['sent'] + s2['dev']['sent'] +
                                         s1['test']['sent']  + s2['test']['sent'])
    
    print('** DICTIONARY : %i words in dictionary' % len(id2word))
    
    for data_type in ['train', 'dev', 'test']:
        s1[data_type]['data'] = np.array([torch.LongTensor([word2id[w] for w in ['<s>'] + s + ['</s>']]) for s in s1[data_type]['sent']])
        s2[data_type]['data'] = np.array([torch.LongTensor([word2id[w] for w in ['<s>'] + s + ['</s>']]) for s in s2[data_type]['sent']])
        assert len(s1[data_type]['data']) == len(s2[data_type]['data'])
        
    train = {'s1':s1['train']['data'], 's2':s2['train']['data'], 'label':target['train']['data']}
    dev = {'s1':s1['dev']['data'], 's2':s2['dev']['data'], 'label':target['dev']['data']}
    test  = {'s1':s1['test']['data'] , 's2':s2['test']['data'] , 'label':target['test']['data'] }
    
    return train, dev, test, id2word, word2id
