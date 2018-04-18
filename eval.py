import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, build_vocab, get_batch
from mutils import get_optimizer
from models import NLINet

GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"

IDX2LBL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def get_args():
  parser = argparse.ArgumentParser(description='Evaluating NLI InferSent model')

  # paths
  parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
  parser.add_argument("--embdfile", type=str, default='../data/embds/glove.840B.300d.txt', help="File containin the word embeddings")
  parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
  parser.add_argument("--model", type=str, help="Input model that has already been trained")
  parser.add_argument("--pred_file", type=str, default='preds', help="Suffix for the prediction files")
  parser.add_argument("--train_lbls_file", type=str, default='../data/snli_1.0/cl_snli_train_lbl_file', help="NLI train data labels file (SNLI or MultiNLI)")
  parser.add_argument("--train_src_file", type=str, default='../data/snli_1.0/cl_snli_train_source_file', help="NLI train data source file (SNLI or MultiNLI)")
  parser.add_argument("--val_lbls_file", type=str, default='../data/snli_1.0/cl_snli_val_lbl_file', help="NLI validation (dev) data labels file (SNLI or MultiNLI)")
  parser.add_argument("--val_src_file", type=str, default='../data/snli_1.0/cl_snli_val_source_file', help="NLI validation (dev) data source file (SNLI or MultiNLI)")
  parser.add_argument("--test_lbls_file", type=str, default='../data/snli_1.0/cl_snli_test_lbl_file', help="NLI test data labels file (SNLI or MultiNLI)")
  parser.add_argument("--test_src_file", type=str, default='../data/snli_1.0/cl_snli_test_source_file', help="NLI test data source file (SNLI or MultiNLI)")


  # data
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")

  # model
  parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
  parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
  parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
  parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
  parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
  parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
  parser.add_argument("--batch_size", type=int, default=64)

  # gpu
  parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")


  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()


  # set gpu device
  torch.cuda.set_device(params.gpu_id)

  # print parameters passed, and all parameters
  print('\ntogrep : {0}\n'.format(sys.argv[1:]))
  print(params)

  return params

def evaluate(epoch, valid, params, word_vec, nli_net, eval_type, pred_file):
  nli_net.eval()
  correct = 0.
  global val_acc_best, lr, stop_training, adam_stop

  #if eval_type == 'valid':
  print('\n{0} : Epoch {1}'.format(eval_type, epoch))


  s1 = valid['s1'] if eval_type == 'valid' else test['s1']
  s2 = valid['s2'] if eval_type == 'valid' else test['s2']
  target = valid['label'] if eval_type == 'valid' else test['label']

  out_preds_f = open(pred_file, "wb")

  for i in range(0, len(s1), params.batch_size):
    # prepare batch
    s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
    s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
    s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
    tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

    # model forward
    output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

    pred = output.data.max(1)[1]
    for p in pred:
      out_preds_f.write(IDX2LBL[pred[0]] + "\n")
    correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

  # save model
  eval_acc = round(100 * correct / len(s1), 2)

  out_preds_f.close()

  print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))

  return eval_acc


def main(args):
  print "main"

  """
  SEED
  """
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  """
  DATA
  """

  train, valid, test = get_nli(args.nlipath)
  word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], GLOVE_PATH)

  for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
      eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

  args.word_emb_dim = 300

  nli_net = torch.load(args.model)
  print(nli_net)

  # loss
  weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn = nn.CrossEntropyLoss(weight=weight)
  loss_fn.size_average = False

  if args.gpu_id > -1:
    nli_net.cuda()
    loss_fn.cuda()

  """
  Train model on Natural Language Inference task
  """
  epoch = 1

  for pair in [(train, 'train'), (valid, 'val'), (test, 'test')]:
    #args.batch_size = len(pair[0]['lbls'])
    eval_acc = evaluate(0, pair[0], args, word_vec, nli_net, pair[1], "%s/%s_%s" % (args.outputdir, pair[1], args.pred_file))
    #epoch, valid, params, word_vec, nli_net, eval_type


if __name__ == '__main__':
  args = get_args()
  main(args)
