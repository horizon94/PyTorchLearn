# adapted from https://github.com/taolei87/rcnn/tree/master/code/rationale

from utils.utils import say
import gzip
import random
import json

import numpy as np
import torch


# this method from https://github.com/taolei87/rcnn/tree/master/code/utils/__init__.py:
def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                # vals = np.array([float(x) for x in parts[1:]])
                vals = torch.FloatTensor([float(x) for x in parts[1:]])
                yield word, vals
class embedding_loader():
    def __init__(self,path):
        self.path=path
        self.word2idx, self.idx2word, self.embeddings=self.load_embedding(self.path)

    def load_embedding(self,path):
        file_open = gzip.open if path.endswith(".gz") else open
        word2idx={}
        idx2word=[]
        embeddings=[]
        with file_open(path) as fin:
            count=0
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    word = parts[0]
                    if word in idx2word:
                        continue
                    word2idx[word] = len(word2idx)
                    idx2word.append(word)
                    # vals = np.array([float(x) for x in parts[1:]])
                    vals = np.array([float(x) for x in parts[1:]])
                    embeddings.append(vals)
                count+=1
                if count%100==0:
                    print(count)
        embeddings = np.vstack(embeddings).astype(np.float32)
        return word2idx, idx2word, embeddings
    def map_words_to_indexes(self,sentence):
        return [self.word2idx[word] for word in sentence]
    def map_indexes_to_words(self,sentence):
        return [self.idx2word[id] for id in sentence]
    def map_word_to_index(self,word):
        return self.word2idx[word]
    def map_index_to_word(self,index):
        return self.idx2word[index]

def read_rationales(path):
    data = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path, 'rt') as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    return data


def read_annotations(path):
    data_x, data_y = [], []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path, 'rt') as fin:
        count=0
        for line in fin:
            # y, sep, x = line.partition("\t")
            y, x, _, _, _ = line.split("\t")
            x, y = x.split(" "), y.split(" ")
            if len(x) == 0:
                continue
            y = np.asarray([float(v) for v in y], dtype=np.float32)
            x = filter(lambda xi: xi != "<padding>", x)
            data_x.append(x)
            data_y.append(y)
            print(count)
            count+=1
    say("{} examples loaded from {}\n".format(len(data_x), path))
    say("max text length: {}\n".format(max(len(x) for x in data_x)))
    return data_x, data_y

def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y = [], []
    N = len(x)
    M = (N - 1) // batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
    for i in range(M):
        bx, by = create_one_batch(
                    lstx=x[i*batch_size:(i+1)*batch_size],
                    lsty=y[i*batch_size:(i+1)*batch_size],
                    padding_id=padding_id
                )
        batches_x.append(bx)
        batches_y.append(by)
    if sort:
        random.seed(5817)
        perm2 = list(range(M))
        random.shuffle(perm2)
        batches_x = [batches_x[i] for i in perm2]
        batches_y = [batches_y[i] for i in perm2]
    return batches_x, batches_y

def create_batches_with_num(x, y, num, batch_size, padding_id, sort=True):
    batches_x, batches_y, batches_num = [ ], [ ], [ ]
    N = len(x)
    M = (N-1)/batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [ x[i] for i in perm ]
        y = [ y[i] for i in perm ]
        num = [num[i] for i in perm]
    for i in xrange(M):
        bx, by= create_one_batch(
                    x[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size],
                    padding_id
                )
        bnum=num[i*batch_size:(i+1)*batch_size]
        batches_x.append(bx)
        batches_y.append(by)
        batches_num.append(bnum)
    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [ batches_x[i] for i in perm2 ]
        batches_y = [ batches_y[i] for i in perm2 ]
        batches_num = [batches_num[i] for i in perm2]
    return batches_x, batches_y, batches_num

def create_one_batch(lstx, lsty, padding_id):
    """
    lstx is a list of 1-d LongTensors
    """
    batch_size = len(lstx)
    max_len = max(x.shape[0] for x in lstx)
    assert min(x.shape[0] for x in lstx) > 0
    bx = torch.LongTensor(max_len, batch_size)
    bx.fill_(padding_id)
    for n in range(batch_size):
        this_len = lstx[n].shape[0]
        bx[:this_len, n] = lstx[n]
    by = torch.Tensor(lsty)
    return bx, by
