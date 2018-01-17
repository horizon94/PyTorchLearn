import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    # hyperparameter
    argparser.add_argument("--class_num",
                           type=int,
                           default=51,
                           help="number of classes")

    argparser.add_argument("--max_epoches",
                           type=int,
                           default=100,
                           help="max epoches")

    argparser.add_argument("--learning_rate",
                           type=float,
                           default=0.1,
                           help="learning rate")

    argparser.add_argument("--rnn_type",
                           type=str,
                           default="gru",
                           help="rnn type")

    argparser.add_argument("--max_len",
                           type=int,
                           default=256,
                           help="max sentence length")
    
    argparser.add_argument("--num_layers",
                           type=int,
                           default=1,
                           help="rnn layer number"
                            )
    argparser.add_argument("--batch_size",
                           type=int,
                           default=64,
                           help="batch size")

    argparser.add_argument("--embedding_dim",
                           type=int,
                           default=200,
                           help="word embedding dimension")

    argparser.add_argument("--hidden_dim",
                           type=int,
                           default=321,
                           help="lstm hidden dimension")

    argparser.add_argument("--loss_func",
                           type=str,
                           default="ce",
                           help="loss function: [ce, mse]")

    argparser.add_argument("--expected_z_size",
                           type=int,
                           default=20,
                           help="expected rationale size")

    argparser.add_argument("--sparsity",
                           type=float,
                           default=0.0003,
                           help="rationale size penalty factor")

    argparser.add_argument("--coherence",
                           type=float,
                           default=2,
                           help="rationale coherence penalty factor")

    # pathes
    argparser.add_argument("--embedding",
                           type=str,
                           default="/home/jx/rcnn/data/word2Vec-200.txt",
                           help="pretrained embedding file path")

    argparser.add_argument("--train",
                           type=str,
                           default="/home/jx/rcnn/data/chunkNew/train.txt",
                           help="train file path")

    argparser.add_argument("--valid",
                           type=str,
                           default="/home/jx/rcnn/data/chunkNew/valid.txt",
                           help="valid file path")

    argparser.add_argument("--test_json",
                           type=str,
                           default="/home/jx/rcnn/data/chunkNew/annotations.chunk.json",
                           help="test_json file path")

    argparser.add_argument("--save_model",
                           type=str,
                           help="save model path")





    argparser.add_argument("--use_gpu",
                           type=bool,
                           default=True,
                           help="use gpu")
    args = argparser.parse_args()
    return args
