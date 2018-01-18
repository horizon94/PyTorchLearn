import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.nn import Parameter


class ZlayerCell(nn.Module):
    def __init__(self, input_x_dim):
        super(ZlayerCell, self).__init__()
        self.input_size = input_x_dim
        self.activation = nn.Sigmoid()
        self.linear = nn.Linear(input_x_dim+1, 1)
        # self.w = Parameter(Variable(torch.FloatTensor(input_x_size, 1)))
        # self.b = Parameter(Variable(torch.FloatTensor(1)))

    def forward(self, x_t, z_tm1):
        # x_t shape (batch_size,hidden_dim)
        # z_tm1 shape (batch_size, 1)
        # x_t = x_t.expand(1, x_t.size()[0], x_t.size()[1])
        # print(x_t.shape)
        # print(z_tm1.shape)
        xz = torch.cat((x_t, z_tm1), 1)
        pz = self.activation(self.linear(xz))  # pz shape(batch_size, 1)
        # pz = pz.expand(1, pz.size()[0], pz.size()[1])  # pz shape(1,batch_size, 1)
        return pz


class Zlayer(nn.Module):
    def __init__(self, args, input_x_dim, batch_size):
        super(Zlayer, self).__init__()
        self.batch_size = batch_size
        self.cell = ZlayerCell(input_x_dim)
        if args.use_gpu and torch.cuda.is_available():
            self.initial_state = Variable(torch.zeros(self.batch_size,1).cuda())
        else:
            self.initial_state = Variable(torch.zeros(self.batch_size, 1))

    def forward(self, input_x):
        # input_x shape (sequence_length,batch_size,hidden_dim)
        outputs = []
        # Use priming string to "build up" hidden state
        pz = self.initial_state
        for p in range(len(input_x)):
            pz = self.cell(input_x[p], pz)  # pz shape(1, batch_size, 1)
            outputs.append(pz)  # outputs sequence_lenth * shape( batch_size, 1)

        outputs = [a.expand(1, a.size()[0], a.size()[1]) for a in outputs]
        # outputs shape=(sequence_len,batch_size,1)
        outputs = torch.cat(outputs, dim=0)
        # outputs shape=(sequence_len,batch_size)
        outputs.squeeze_()
        return outputs

    # def init_hidden(self):
    #     return Variable(torch.zeros(self.batch_size,1))#requires_grad=False

if __name__=='__main__':
    model=Zlayer(30, 64)
    x=Variable(torch.FloatTensor(64, 100, 30)).t()
    pz=model(x)
    print(pz.shape)
