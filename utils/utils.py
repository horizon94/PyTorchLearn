#-*- coding: UTF-8 -*-
import sys
import gzip

import numpy as np

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                #print len(parts)
                word = parts[0]
                #drint word
                vals = np.array([ float(x) for x in parts[1:] ])
                #print "vect="+str(vals)
                yield word, vals

