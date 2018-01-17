import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--class_num",
                           type=int,
                           default=51,
                           help="number of classes"
                           )
    argparser.add_argument("--batch_size",
                           type=int,
                           default=64,
                           help="batch size"
                           )
    argparser.add_argument("--embed_dim",
                           type=int,
                           default=2,
                           help="word embedding dimension"
                           )
    argparser.add_argument("--hidden_dim",
                           type=int,
                           default=321,
                           help="lstm hidden dimension")
