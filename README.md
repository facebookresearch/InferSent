# InferSent

*InferSent* encoder and training code from the paper [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364).

## Dependencies

This code is written in python. The dependencies are :

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12
‡

## Download datasets
To get all the SNLI/MultiNLI datasets, run (in dataset/) :
```bash
./get_data.bash
```
This will automatically download and preprocess SNLI/MultiNLI, and put them in data/senteval_data.
It will also put GloVe vectors in that directory (needed for the model).


## Use InferSent sentence encoder
0) Download and load our model
```bash
curl -o encoder/infersent.pickle https://s3.amazonaws.com/infersent/infersent.pickle
```

1) Load our pre-trained infersent model (in encoder/) :
```python
import torch
infersent = torch.load('infersent.pickle')
```
Note : you need the file "models.py" that provides the definition of the model to be able to load it.

2) Set GloVe path for the model : 
```python
infersent.set_glove_path(glove_path)
```
where glove_path is the path to the file *'glove.840B.300d.txt'* of glove vectors with which our model was trained. Note that using [GloVe](https://nlp.stanford.edu/projects/glove/) vectors allows to have a coverage of more than 2 million english words.


3) Build the vocabulary of glove vectors (and keep only those needed) : 
```python
infersent.build_vocab(sentences1, tokenize=True)
```
where *sentences* (required) is your list of *n* sentences. You can also update your vocabulary with new words using *infersent.update_vocab(sentences, glove_path, tokenize=True)*. 

If your sentences are not tokenized, the *tokenize* option (True by default) will use *NLTK 3* to preprocess it.

4) Encode your sentences :
```python
infersent.encode(sentences1, tokenize=True)
```
This will output an numpy array with *n* vectors of dimension **4096** (dimension of the infersent embeddings), which are our general-purpose sentence embeddings.

5) Visualize the value that infersent attributes to each word :
```python
infersent.visualize('The dog is in the kitchen.', tokenize=True)
```



## Train model



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