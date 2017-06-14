# InferSent
*InferSent* is a sentence embeddings method learned from natural language inference data 

*InferSent* encoder and training code from the paper [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364).

In this repo, we provide our pre-trained sentence encoder that outperforms previous approaches when used as features for many different tasks. See [SentEval](https://github.com/aconneau/SentEval) for our sentence embeddings evaluation tool.

## Dependencies

This code is written in python. The dependencies are :

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12

## Download datasets
To get the SNLI and MultiNLI datasets (and GloVe), run (in dataset/) :
```bash
./get_data.bash
```
This will download and preprocess SNLI/MultiNLI, and put them in data/senteval_data, same for GloVe.


## Use InferSent sentence encoder
0) Download and load our model
```bash
curl -Lo encoder/infersent.pickle https://s3.amazonaws.com/senteval/infersent/infersent.pickle
```

1) Load our pre-trained infersent model (in encoder/) :
```python
import torch
infersent = torch.load('infersent.pickle')
```
Note : you need the file "models.py" that provides the definition of the model to load it.

2) Set GloVe path for the model : 
```python
infersent.set_glove_path(glove_path)
```
where glove_path is the path to the file *'glove.840B.300d.txt'* of glove vectors with which our model was trained. Note that using [GloVe](https://nlp.stanford.edu/projects/glove/) vectors allows to have a coverage of more than 2 million english words.


3) Build the vocabulary of glove vectors (i.e keep only those needed) : 
```python
infersent.build_vocab(sentences, tokenize=True)
```
where *sentences* (required) is your list of *n* sentences. You can also update your vocabulary with new words using *infersent.update_vocab(sentences)*.

If your sentences are not tokenized, the *tokenize* option (default: True) will use *NLTK 3* to preprocess it.

4) Encode your sentences :
```python
infersent.encode(sentences, tokenize=True)
```
This will output an numpy array with *n* vectors of dimension **4096** (dimension of the infersent embeddings), which are our general-purpose sentence embeddings.

5) Visualize the value that infersent attributes to each word (~hidden state h_t of the BiLSTM) :
```python
infersent.visualize('A man playing an instrument.', tokenize=True)
```

![Model](https://s3.amazonaws.com/senteval/infersent/visualization.png)


## Train model on SNLI
To reproduce our results on [SNLI](https://nlp.stanford.edu/projects/snli/), set **GLOVE_PATH** in *train_nli.py* and run:
```bash
python train_nli.py
```
You should obtain a test accuracy around [84.5](https://nlp.stanford.edu/projects/snli/) with our BiLSTM-max.

## Reproduce our results on transfer tasks
To reproduce our results on transfer tasks, you need to clone [SentEval](https://github.com/aconneau/SentEval) and 

Clone [SentEval](https://github.com/aconneau/SentEval) and set **PATH_SENTEVAL**, **PATH_TRANSFER_TASKS** in *evaluate_model.py*.
```bash
python evaluate_model.py
```

## References

Please cite [1](https://arxiv.org/abs/1705.02364) if using this code for evaluating sentence embedding methods.

### Supervised Learning of Universal Sentence Representations from Natural Language Inference Data

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

```
@article{conneau2017supervised,
  title={Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
  journal={arXiv preprint arXiv:1705.02364},
  year={2017}
}
```