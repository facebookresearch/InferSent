import re
import inspect
import torch
from torch import optim
from torch.autograd import Variable
from data import get_dict, get_lut_glove, get_batch

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params

"""
Importing batcher and prepare for SentEval
"""

def batcher(network, batch, params):
    # batch contains list of words
    X = [torch.LongTensor([params.word2id[w] if w in params.word2id else params.word2id['<p>'] for w in ['<s>'] + s + ['</s>']]) for s in batch]
    X, X_len = get_batch(X, params.word2id['<p>'])
    
    k = X.size(1)  # actual batch size

    # forward
    X = Variable(X, volatile=True).cuda()
    X_embed = params.lut(X)
    
    embeddings = network.encode((X_embed, X_len))
    
    return embeddings.data.cpu().numpy()

def prepare(params, samples):
    _, params.word2id = get_dict(samples)
    params.emb_dim = 300
    params.eos_index = params.word2id['</s>']
    params.sos_index = params.word2id['</s>']
    params.pad_index = params.word2id['<p>']
    _, params.lut, _ = get_lut_glove('840B.300d', params.word2id)
    params.lut.cuda()
    return # prepare puts all of its outputs in params.{}

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__