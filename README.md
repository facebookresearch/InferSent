# InferSent

*InferSent* is a *sentence embeddings* method that provides semantic sentence representations. It is trained on natural language inference data and generalizes well to many different tasks.

We provide our pre-trained sentence encoder for reproducing the results from [our paper](https://arxiv.org/abs/1705.02364). See also [SentEval](https://github.com/facebookresearch/SentEval) for automatic evaluation of the quality of sentence embeddings.

## Dependencies

This code is written in python. The dependencies are:

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12
* NLTK >= 3

## Download datasets
To get GloVe, SNLI and MultiNLI [2GB, 90MB, 216MB], run (in dataset/):
```bash
./get_data.bash
```
This will download GloVe and preprocess SNLI/MultiNLI datasets. For MacOS, you may have to use *p7zip* instead of *unzip*.


## Use our sentence encoder
We provide a simple interface to encode english sentences. **See [**encoder/demo.ipynb**](https://github.com/facebookresearch/InferSent/blob/master/encoder/demo.ipynb)
for a practical example.** Get started with the following steps:

*0.0) Download our model trained on AllNLI (SNLI and MultiNLI) [147MB]:*
```bash
curl -Lo encoder/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle
```

*0.1) Make sure you have the NLTK tokenizer by running the following once:*
```python
import nltk
nltk.download('punkt')
```

*1) Load our pre-trained model (in encoder/):*
```python
import torch
# if you are on GPU (encoding ~1000 sentences/s, default)
infersent = torch.load('infersent.allnli.pickle')
# if you are on CPU (~40 sentences/s)
infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
```
Note: To load the model, you need "encoder/models.py" in your working directory.

*2) Set GloVe path for the model:*
```python
infersent.set_glove_path(glove_path)
```
where *glove_path* is the path to *'glove.840B.300d.txt'*, containing glove vectors with which our model was trained. Note that using [GloVe](https://nlp.stanford.edu/projects/glove/) vectors allows to have a coverage of more than *2 million* english words.


*3) Build the vocabulary of word vectors (i.e keep only those needed):*
```python
infersent.build_vocab(sentences, tokenize=True)
```
where *sentences* is your list of **n** sentences. You can update your vocabulary using *infersent.update_vocab(sentences)*, or directly load the **K** most common english words with *infersent.build_vocab_k_words(K=100000)*.
If **tokenize** is True (by default), sentences will be tokenized using NTLK.

*4) Encode your sentences (list of *n* sentences):*
```python
infersent.encode(sentences, tokenize=True)
```
This will output an numpy array with *n* vectors of dimension **4096** (dimension of the sentence embeddings). Speed is around *1000 sentences per second* with batch size 128 on a single GPU.

*5) Visualize the importance that our model attributes to each word:*

Our representations were trained to focus on semantic information such that a classifier can easily tell the difference between contradictory, neutral or entailed sentences. We provide a function to visualize the importance of each word in the encoding of a sentence:
```python
infersent.visualize('A man plays an instrument.', tokenize=True)
```
![Model](https://s3.amazonaws.com/senteval/infersent/visualization.png)


## Train model on Natural Language Inference (SNLI)
To reproduce our results and train our models on [SNLI](https://nlp.stanford.edu/projects/snli/), set **GLOVE_PATH** in *train_nli.py*, then run:
```bash
python train_nli.py
```
You should obtain a dev accuracy of 85 and a test accuracy of **[84.5](https://nlp.stanford.edu/projects/snli/)** with the default setting.

## Reproduce our results on transfer tasks
To reproduce our results on transfer tasks, clone [SentEval](https://github.com/facebookresearch/SentEval) and set **PATH_SENTEVAL**, **PATH_TRANSFER_TASKS** in *evaluate_model.py*, then run:
```bash
python evaluate_model.py
```

Using our best model *infersent.allnli.pickle*, you should obtain the following test results:

Model | MR | CR | SUBJ | MPQA | STS14 | [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results) | SICK Relatedness | SICK Entailment | SST | TREC | MRPC
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
**`InferSent`** | **81.1** | **86.3** | 92.4 | **90.2** | **.68/.65** | **75.8/75.5** | **0.884** | **86.1** | **84.6** | 88.2 | 76.2/83.1
`SkipThought` | 79.4 | 83.1 | **93.7** | 89.3 | .44/.45 | 72.1/70.2| 0.858 | 79.5 | 82.9 | 88.4 | - 

Note that while InferSent provides good features for many different tasks, our approach also obtains strong results on STS tasks which evaluate the quality of the cosine metrics in the embedding space.

## Reference

Please cite [1](https://arxiv.org/abs/1705.02364) if you found this code useful.

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

Contact: [aconneau@fb.com](mailto:aconneau@fb.com)

