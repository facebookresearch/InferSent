# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn


"""
InferSent encoder
""" 
class BLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BLSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.cuda = config['cuda']
        
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=True, dropout=self.dpout_model)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize, self.enc_lstm_dim).zero_()).cuda()
        if self.cuda:
            self.init_lstm = self.init_lstm.cuda()
        
    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch) | sent Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple        
        
        if sent.size(1) != self.init_lstm.size(1):
            self.init_lstm = Variable(torch.FloatTensor(2, sent.size(1), self.enc_lstm_dim).zero_())
            if self.cuda:
                self.init_lstm = self.init_lstm.cuda()
        
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)
        
        idx_sort = torch.cuda.LongTensor(idx_sort) if self.cuda else torch.LongTensor(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))
        
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed, (self.init_lstm, self.init_lstm))[0] #seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        
        # Un-sort by length
        idx_unsort = torch.cuda.LongTensor(idx_unsort) if self.cuda else torch.LongTensor(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))
        
        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path
    
    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize: from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        return word_dict
    
    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), 'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print 'Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict))
        return word_vec
    
    
    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print 'Vocab size : {0}'.format(len(self.word_vec))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need to set_glove_path(glove_path)'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)
        
        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]
                
        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_glove(word_dict)
            self.word_vec.update(new_word_vec)
        print 'New vocab size : {0} (added {1} words)'.format(len(self.word_vec), len(new_word_vec))

    
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))
        
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
        
        return torch.FloatTensor(embed)
    
    
    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        if tokenize: from nltk.tokenize import word_tokenize
        sentences = [['<s>']+s.split()+['</s>'] if not tokenize else ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])
        
        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word if word in self.word_vec else '<p>' for word in sentences[i]]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print 'Nb words kept : {0}/{1} ({2} %)'.format(n_wk, n_w, round((100.0 * n_wk) / n_w, 2))
                                                  
        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]
        
        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(sentences[stidx:stidx + bsize]), volatile=True)
            if self.cuda:
                batch = batch.cuda()
            batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)
        
        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]
        
        if verbose:
            print 'Speed : {0} sentences/s ({1} mode)'.format(round(len(embeddings)/(time.time()-tic), 2),\
                                                              'gpu' if self.cuda else 'cpu')
        return embeddings
    
    def visualize(self, sent, tokenize=True):
        if tokenize: from nltk.tokenize import word_tokenize
        
        sent = sent.split() if not tokenize else word_tokenize(sent)
        sent = [['<s>'] + [word for word in sent if word in self.word_vec] + ['</s>']]
        print sent
        if ' '.join(sent[0]) == '<s> </s>':
            import warnings
            warnings.warn('No words in "{0}" (idx={1}) have glove vectors. Replacing by "<s> </s>"..'.format(sentences[i], i))   
        batch = Variable(self.get_batch(sent), volatile=True)
        
        init_lstm = Variable(torch.FloatTensor(2, 1, self.enc_lstm_dim).zero_())
        if self.cuda:
            init_lstm = init_lstm.cuda()
            batch = batch.cuda()
        output = self.enc_lstm(batch, (init_lstm, init_lstm))[0]
        output, idxs = torch.max(output, 0)
        #output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs==k)) for k in range(len(sent[0]))]
        
        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
        fig = plt.figure()
        plt.xticks(x, sent[0])
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()
        
        return output, idxs
                             
