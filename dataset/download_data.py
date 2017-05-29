import os
import argparse

parser = argparse.ArgumentParser(description='get all data for infersent')
parser.add_argument('--config_path', required=False,
                    help='path', default='.')
args = parser.parse_args()

data_path = args.config_path
os.system('mkdir ' + data_path)
preprocess_exec = './tokenizer.sed'



snlipath = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
multinlipath = 'https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
glovepath = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'



# GloVe
os.system('mkdir ' + os.path.join(data_path, 'GloVe'))
os.system('wget -O - ' + glovepath + ' > ' + os.path.join(data_path, 'GloVe', 'glove.840B.300d.zip'))
os.system('unzip ' + os.path.join(data_path, 'GloVe', 'glove.840B.300d.zip') + ' -d ' + os.path.join(data_path, 'GloVe'))
os.system('rm ' + os.path.join(data_path, 'GloVe', 'glove.840B.300d.zip'))


# SNLI
os.system('mkdir ' + os.path.join(data_path, 'SNLI'))
os.system('wget -O - ' + snlipath + ' > ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip'))
os.system('unzip ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip') + ' -d ' + os.path.join(data_path, 'SNLI'))
os.system('rm ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip'))
os.system('rm -r ' + os.path.join(data_path, 'SNLI', '__MACOSX'))


os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_train.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'train.snli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_dev.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'dev.snli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_test.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'test.snli.txt'))

for split in ["train", "dev", "test"]:
    fpath = os.path.join(data_path, 'SNLI', split + '.snli.txt')
    os.system('cut -f1 ' + fpath + ' > ' + os.path.join(data_path, 'SNLI', 'labels.' + split))
    os.system('cut -f2 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SNLI', 's1.' + split))
    os.system('cut -f3 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SNLI', 's2.' + split))
    os.system('rm ' + fpath)
    os.system('rm -r ' + os.path.join(data_path, 'SNLI', 'snli_1.0'))


# MultiNLI
# Test set not available yet : dev set is "matched" test set is "mismatched"
os.system('mkdir ' + os.path.join(data_path, 'MultiNLI'))
os.system('wget -O - ' + multinlipath + ' > ' + os.path.join(data_path, 'MultiNLI', 'multinli_0.9.zip'))
os.system('unzip ' + os.path.join(data_path, 'MultiNLI', 'multinli_0.9.zip') + ' -d ' + os.path.join(data_path, 'MultiNLI'))
os.system('rm ' + os.path.join(data_path, 'MultiNLI', 'multinli_0.9.zip'))
os.system('rm -r ' + os.path.join(data_path, 'MultiNLI', '__MACOSX'))


os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'MultiNLI', 'multinli_0.9', 'multinli_0.9_train.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'MultiNLI', 'train.multinli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'MultiNLI', 'multinli_0.9', 'multinli_0.9_dev_matched.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'MultiNLI', 'dev.multinli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'MultiNLI', 'multinli_0.9', 'multinli_0.9_dev_mismatched.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'MultiNLI', 'test.multinli.txt'))

for split in ["train", "dev", "test"]:
    fpath = os.path.join(data_path, 'MultiNLI', split + '.multinli.txt')
    os.system('cut -f1 ' + fpath + ' > ' + os.path.join(data_path, 'MultiNLI', 'labels.' + split))
    os.system('cut -f2 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MultiNLI', 's1.' + split))
    os.system('cut -f3 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MultiNLI', 's2.' + split))
    os.system('rm ' + fpath)
os.system('rm -r ' + os.path.join(data_path, 'MultiNLI', 'multinli_0.9'))