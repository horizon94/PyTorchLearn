import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMClassifier(nn.Module):
    def __init__(self,opt):
        super(LSTMClassifier, self).__init__()
        self.opt=opt
        self.embed_dim=opt.embed_dim
        self.hidden_dim=opt.hidden_dim
        self.batch_size=opt.batch_size
        self.use_gpu=torch.cuda.is_available()

        self.embeddingLayer=nn.Embedding(opt.vocab_size,self.embed_dim)
        self.lstm=nn.LSTM(self.embed_dim,self.hidden_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))