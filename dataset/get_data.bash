# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

preprocess_exec=./tokenizer.sed

SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
MultiNLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'



# GloVe
echo $glovepath
mkdir GloVe
curl -LO $glovepath
unzip glove.840B.300d.zip -d GloVe/
rm glove.840B.300d.zip



### download SNLI
mkdir SNLI
curl -o SNLI/snli_1.0.zip $SNLI
unzip SNLI/snli_1.0.zip -d SNLI
rm SNLI/snli_1.0.zip
rm -r SNLI/__MACOSX

for split in train dev test
do
    fpath=SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > SNLI/labels.$split
    cut -f2 $fpath | $preprocess_exec > SNLI/s1.$split
    cut -f3 $fpath | $preprocess_exec > SNLI/s2.$split
    rm $fpath
done
rm -r SNLI/snli_1.0


# MultiNLI
# Test set not available yet : we define dev set as the "matched" set and the test set as the "mismatched"
mkdir MultiNLI
curl -o MultiNLI/multinli_0.9.zip $MultiNLI
unzip MultiNLI/multinli_0.9.zip -d MultiNLI
rm MultiNLI/multinli_0.9.zip
rm -r MultiNLI/__MACOSX


mv MultiNLI/multinli_0.9/multinli_0.9_train.txt MultiNLI/train.multinli.txt
mv MultiNLI/multinli_0.9/multinli_0.9_dev_matched.txt MultiNLI/dev.multinli.txt
mv MultiNLI/multinli_0.9/multinli_0.9_dev_mismatched.txt MultiNLI/test.multinli.txt

rm -r MultiNLI/multinli_0.9

for split in train dev test
do
    fpath=MultiNLI/$split.multinli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' $fpath | cut -f 1,6,7 | sed '1d' > $fpath.tok
    cut -f1 $fpath.tok > MultiNLI/labels.$split
    cut -f2 $fpath.tok | $preprocess_exec > MultiNLI/s1.$split
    cut -f3 $fpath.tok | $preprocess_exec > MultiNLI/s2.$split
    rm $fpath $fpath.tok 
done

