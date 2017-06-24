# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

import argparse
import sys

import numpy as np
import torch

from mutils import dotdict


GLOVE_PATH = "../dataset/GloVe/glove.840B.300d.txt"

PATH_SENTEVAL = "/home/aconneau/notebooks/senteval/"
PATH_TRANSFER_TASKS = "/home/aconneau/notebooks/senteval/data/senteval_data/"

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--modelpath", type=str, default='infersent.pickle', help="path to model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
params, _ = parser.parse_known_args()


# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval    
    

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print sys.argv[1:]
print params


def batcher(batch, params):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)  
    

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',\
                  'SICKEntailment', 'MRPC', 'STS14']

# define senteval params
params_senteval = dotdict({'usepytorch': True,
                           'task_path': PATH_TRANSFER_TASKS,
                           })

# Load model
params_senteval.infersent = torch.load(params.modelpath)
params_senteval.infersent.set_glove_path(GLOVE_PATH)

se = senteval.SentEval(batcher, prepare, params_senteval)
results_transfer = se.eval(transfer_tasks)

print results_transfer

