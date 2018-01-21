import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter


class LSTMClassifier(nn.Module):

    def __init__(self, args, embedding_loader = None):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.use_gpu = args.use_gpu
        self.embedding_loader = embedding_loader
        self.word_embeddings = nn.Embedding(embedding_loader.embeddings.shape[0], args.embedding_dim)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim)
        self.hidden2label = nn.Linear(args.hidden_dim, args.articles_num)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            self.h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            self.c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            self.h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            self.c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (self.h0, self.c0)

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        x = embeds#.view(len(sentence), self.batch_size, -1)
        lstm_out, (h_n,c_n) = self.lstm(x, self.hidden)
        # print(lstm_out.shape,h_n.shape,c_n.shape)
        # print(lstm_out[-1][0])
        # print(h_n[0][0])
        # print("*"*50)
        y  = self.hidden2label(h_n[-1])
        return y
