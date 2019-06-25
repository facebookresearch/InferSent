# InferSent

*InferSent* is a *sentence embeddings* method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks.

We provide our pre-trained English sentence encoder from [our paper](https://arxiv.org/abs/1705.02364) and our [SentEval](https://github.com/facebookresearch/SentEval) evaluation toolkit.

**Recent changes**: Removed train_nli.py and only kept pretrained models for simplicity. Reason is I do not have time anymore to maintain the repo beyond simple scripts to get sentence embeddings.

## Dependencies

This code is written in python. Dependencies include:

* Python 2/3
* [Pytorch](http://pytorch.org/) (recent version)
* NLTK >= 3

## Download word vectors

Download [GloVe](https://nlp.stanford.edu/projects/glove/) (V1) or [fastText](https://fasttext.cc/docs/en/english-vectors.html) (V2) vectors:
```bash
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/
```

## Use our sentence encoder
We provide a simple interface to encode English sentences. **See [**demo.ipynb**](https://github.com/facebookresearch/InferSent/blob/master/demo.ipynb)
for a practical example.** Get started with the following steps:

*0.0) Download our InferSent models (V1 trained with GloVe, V2 trained with fastText)[147MB]:*
```bash
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
```
Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer) and infersent2 is trained with fastText (which have been trained on text preprocessed with the MOSES tokenizer). The latter also removes the padding of zeros with max-pooling which was inconvenient when embedding sentences outside of their batches.

*0.1) Make sure you have the NLTK tokenizer by running the following once:*
```python
import nltk
nltk.download('punkt')
```

*1) [Load our pre-trained model](https://github.com/facebookresearch/InferSent/blob/master/encoder/demo.ipynb) (in encoder/):*
```python
from models import InferSent
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
```

*2) Set word vector path for the model:*
```python
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)
```

*3) Build the vocabulary of word vectors (i.e keep only those needed):*
```python
infersent.build_vocab(sentences, tokenize=True)
```
where *sentences* is your list of **n** sentences. You can update your vocabulary using *infersent.update_vocab(sentences)*, or directly load the **K** most common English words with *infersent.build_vocab_k_words(K=100000)*.
If **tokenize** is True (by default), sentences will be tokenized using NTLK.

*4) Encode your sentences (list of *n* sentences):*
```python
embeddings = infersent.encode(sentences, tokenize=True)
```
This outputs a numpy array with *n* vectors of dimension **4096**. Speed is around *1000 sentences per second* with batch size 128 on a single GPU.

*5) Visualize the importance that our model attributes to each word:*

We provide a function to visualize the importance of each word in the encoding of a sentence:
```python
infersent.visualize('A man plays an instrument.', tokenize=True)
```
![Model](https://dl.fbaipublicfiles.com/infersent/visualization.png)

## Evaluate the encoder on transfer tasks
To evaluate the model on transfer tasks, see [SentEval](https://github.com/facebookresearch/SentEval/tree/master/examples). Be mindful to choose the same tokenization used for training the encoder. You should obtain the following test results for the baselines and the InferSent models:

Model | MR | CR | SUBJ | MPQA | STS14 | [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results) | SICK Relatedness | SICK Entailment | SST | TREC | MRPC
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
`InferSent1` | **81.1** | **86.3** | 92.4 | **90.2** | **.68/.65** | 75.8/75.5 | 0.884 | 86.1 | **84.6** | 88.2 | **76.2**/83.1
`InferSent2` | 79.7 | 84.2 | 92.7 | 89.4 | **.68/.66** | **78.4/78.4** | **0.888** | **86.3** | 84.3 | **90.8** | 76.0/**83.8**
`SkipThought` | 79.4 | 83.1 | **93.7** | 89.3 | .44/.45 | 72.1/70.2| 0.858 | 79.5 | 82.9 | 88.4 | -
`fastText-BoV` | 78.2 | 80.2 | 91.8 | 88.0 | .65/.63 | 70.2/68.3 | 0.823 | 78.9 | 82.3 | 83.4 | 74.4/82.4

## Reference

Please consider citing [[1]](https://arxiv.org/abs/1705.02364) if you found this code useful.

### Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (EMNLP 2017)

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

```
@InProceedings{conneau-EtAl:2017:EMNLP2017,
  author    = {Conneau, Alexis  and  Kiela, Douwe  and  Schwenk, Holger  and  Barrault, Lo\"{i}c  and  Bordes, Antoine},
  title     = {Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {670--680},
  url       = {https://www.aclweb.org/anthology/D17-1070}
}
```

### Related work
* [J. R Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun, S. Fidler - SkipThought Vectors, NIPS 2015](https://arxiv.org/abs/1506.06726)
* [S. Arora, Y. Liang, T. Ma - A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017](https://openreview.net/pdf?id=SyK00v5xx)
* [Y. Adi, E. Kermany, Y. Belinkov, O. Lavi, Y. Goldberg - Fine-grained analysis of sentence embeddings using auxiliary prediction tasks, ICLR 2017](https://arxiv.org/abs/1608.04207)
* [A. Conneau, D. Kiela - SentEval: An Evaluation Toolkit for Universal Sentence Representations, LREC 2018](https://arxiv.org/abs/1803.05449)
* [S. Subramanian, A. Trischler, Y. Bengio, C. J Pal - Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, ICLR 2018](https://arxiv.org/abs/1804.00079)
* [A. Nie, E. D. Bennett, N. D. Goodman - DisSent: Sentence Representation Learning from Explicit Discourse Relations, 2018](https://arxiv.org/abs/1710.04334)
* [D. Cer, Y. Yang, S. Kong, N. Hua, N. Limtiaco, R. St. John, N. Constant, M. Guajardo-Cespedes, S. Yuan, C. Tar, Y. Sung, B. Strope, R. Kurzweil - Universal Sentence Encoder, 2018](https://arxiv.org/abs/1803.11175)
* [A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni - What you can cram into a single vector: Probing sentence embeddings for linguistic properties, ACL 2018](https://arxiv.org/abs/1805.01070)
* [A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, S. Bowman - GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
