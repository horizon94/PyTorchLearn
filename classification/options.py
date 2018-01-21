import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--class_num",
	    type = int,
	    default = 51,
	    help = "number of classes"
	)
    argparser.add_argument("--word_embedding",
            type = int,
            default = 2,
            help = "word embedding dimension"
        )
    argparser.add_argument("--hidden",
	    type = int,
	    default = 321,
	    help = "lstm hidden dimension")
