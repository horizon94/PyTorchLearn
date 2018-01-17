import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.nn import Parameter
from nn.ZLayer import Zlayer
import rationale.options as options
import rationale.myio as myio
import time
from sklearn import metrics
from utils.utils import say
class Generator(nn.Module):
    def __init__(self,args,embedding):
        super(Generator, self).__init__()
        self.embedding = embedding  # nn.Embedding(args.vocab_size,args.embedding_dim)
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.sentence_length = args.sentence_length
        if args.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size = args.embedding_dim,
                hidden_size = args.hidden_dim,
                bidirectional=True)
            self.initial_state = Variable(torch.zeros(self.batch_size, self.hidden_dim)), \
                                 Variable(torch.zeros(self.batch_size, self.hidden_dim))
        elif args.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size = args.embedding_dim,
                hidden_size = args.hidden_dim,
                bidirectional=True)
            self.initial_state = Variable(torch.zeros(self.batch_size, self.hidden_dim))
        self.zlayer = Zlayer(args.hidden_dim, args.batch_size)


    def forward(self,sentences):
        # sentences shape(sentence_length, batch_size)
        # embeddings shape(sentence_length, batch_size, embedding_dim)
        embeddings=self.embedding(sentences)
        # hidden_states shape(sentence_length, batch_size, hidden_dim)
        hidden_states, _ = self.rnn(embeddings,self.initial_state)
        # pz shape(sentence_length,batch_size)
        pz = self.zlayer(hidden_states)
        # z shape(sentence_length,batch_size)
        z_bernoulli = torch.distributions.Bernoulli(pz)
        z = z_bernoulli.sample()
        # z_size shape (batch_size)
        z_sizes = z.sum(dim=0).int()
        # max_z_sizes = z_sizes.max()
        #rationales shape(sentence_length, batch_size)
        rationales = torch.LongTensor(self.sentence_length, self.batch_size)
        rationales.fill_(self.pad_id)
        for n in range(self.batch_size):
            this_len = z_sizes[n].data[0]
            rationales[:this_len, n] = torch.masked_select(
                sentences[:, n].data, z[:, n].data
            )
        return pz, z, rationales, z_sizes, z_bernoulli



    # def initial_state(self):
    #     self.h0 = Variable(torch.zeros(self.batch_size, self.hidden_dim))
    #     self.c0 = Variable(torch.zeros(self.batch_size, self.hidden_dim))
    #     return self.h0,self.c0

class Encoder(nn.Module):
    def __init__(self, args, embedding):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.class_num = args.class_num
        if args.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_dim,
                num_layers=self.num_layers)
            self.initial_state = Variable(torch.zeros(self.batch_size, self.hidden_dim)), \
                                 Variable(torch.zeros(self.batch_size, self.hidden_dim))
        elif args.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_dim,
                num_layers=self.num_layers)
            self.initial_state = Variable(torch.zeros(self.batch_size, self.hidden_dim))
        self.linear = nn.Linear(self.num_hidden, args.class_num)

    def forward(self, x):
        # x shape (sentence_length, batch_size)
        # outputs shape (sentence_length, batch_size, hidden_dim)
        outputs, _ = self.rnn(x,self.initial_state)
        # final_embed shape (batch_size, hidden_dim)
        final_embed = torch.mean(outputs, dim=0)
        # result shape (batch_size, class_num)
        scores = self.linear(final_embed)
        return scores

class Model(nn.Module):
    def __init__(self, args=args, embedding_loader=None):
        super(Model, self).__init__()
        self.embedding_loader=embedding_loader
        self.embedding = nn.Embedding(args.vocab_size,args.embedding_dim)
        if args.embedding and embedding_loader is not None:
            self.embedding.weight=nn.Parameter(torch.FloatTensor(embedding_loader.embeddings))
        self.generator = Generator(args,self.embedding)
        self.encoder = Encoder(args,self.embedding)

    def forward(self, x):
        self.pz, self.z, self.rationales, self.z_sizes, self.z_bernoulli = self.generator(x)
        self.scores = self.encoder(self.rationales)
        return self.scores, self.pz, self.z, self.rationales, self.z_sizes, self.z_bernoulli

    def save_model(self, path):
        torch.save(self.state_dict(), path + '/params')
        with open(path+ '/args') as f:
            f.write(str(args))



def train(args, model,train_data, valid_data, test_data = None):
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
    padding_id=model.embedding_loader.map_word_to_index("<padding>")
    if valid_data is not None:
        valid_batches_x, valid_batches_y = myio.create_batches(
            valid_data[0], valid_data[1], args.batch_size, padding_id
        )
    if test_data is not None:
        test_batches_x, test_batches_y, test_batches_num = myio.create_batches_with_num(
            [i["xids"] for i in test_data],
            [i["y"] for i in test_data],
            [i["number"] for i in test_data],
            args.batch_size,
            padding_id,
            sort=False
        )
    best_valid_Loss = 1e100
    for epoch in range(args.max_epoches):
        train_batches_x, train_batches_y = myio.create_batches(
            train_data[0], train_data[1], args.batch_size, padding_id
        )
        epoch_start_time = time.time()
        train_epoch_loss = 0.
        batch_num = len(train_batches_x)
        for b in range(batch_num):
            bx = torch.FloatTensor(train_batches_x[b])
            by = torch.LongTensor(train_batches_y[b])
            if args.use_gpu and torch.cuda.is_available():
                bx, by = Variable(bx.cuda()), Variable(by.cuda())
            else:
                bx, by = Variable(bx), Variable(by)
            model.zero_grad()
            scores, pz, z, rationales, z_sizes, z_bernoulli = model(bx)
            if args.loss_func is "mse":
                pred_loss = nn.MSELoss(nn.Sigmoid(scores),by)
            elif args.loss_func is "ce":
                pred_loss = nn.CrossEntropyLoss(scores,by)
            z_size_loss = torch.abs(z_sizes.sum()-args.expected_z_size)
            z_coh_loss = (z[1:,:] - z[:-1,:]).abs().sum().float()
            reward = -(pred_loss + args.sparsity * z_size_loss + args.coherence * z_coh_loss)
            loss = -z_bernoulli.log_prob(z) * reward
            loss.backward()
            optimizer.step()
            train_epoch_loss += pred_loss.data[0]
        epoch_train_time = time.time() - epoch_start_time
        valid_loss = valid(args, model, valid_batches_x, valid_batches_y)
        say('epoch %s train loss %.3f traintime %s validate loss %.3f\n' % (
            epoch, train_epoch_loss / batch_num, int(epoch_train_time), valid_loss))
        # print('    validate loss %.3f' % (epoch_loss / num_batches))
        if valid_loss < best_valid_Loss:
            bestvalid_loss = valid_loss
            say("BETTER MODEL! SAVING...")
            model.save_model(args.save_model)
            say("\n")
            test(args, model, test_batches_x, test_batches_y, test_batches_num)



def valid(args, model,valid_batches_x, valid_batches_y):
    valid_batch_num = len(valid_batches_x)
    valid_total_loss = 0.
    for b in range(valid_batch_num):
        print('.', end='', flush=True)
        bx = torch.FloatTensor(valid_batches_x[b])
        by = torch.LongTensor(valid_batches_y[b])
        if args.use_gpu and torch.cuda.is_available():
            bx, by = Variable(bx.cuda()), Variable(by.cuda())
        else:
            bx, by = Variable(bx), Variable(by)
        scores, pz, z, rationales, z_sizes, z_bernoulli = model(bx)
        if args.loss_func is "mse":
            pred_loss = nn.MSELoss(nn.Sigmoid(scores), by)
        elif args.loss_func is "ce":
            pred_loss = nn.CrossEntropyLoss(scores, by)
        valid_total_loss += pred_loss.data[0]
    return valid_total_loss / valid_batch_num


def test(args, model, test_batches_x, test_batches_y, test_batches_num):
    valid_batch_num = len(test_batches_x)
    valid_total_loss = 0.
    for b in range(valid_batch_num):
        print('.', end='', flush=True)
        bx = torch.FloatTensor(test_batches_x[b])
        by = torch.LongTensor(test_batches_y[b])
        bnumber = test_batches_num[b]
        if args.use_gpu and torch.cuda.is_available():
            bx, by = Variable(bx.cuda()), Variable(by.cuda())
        else:
            bx, by = Variable(bx), Variable(by)
        scores, pz, z, rationales, z_sizes, z_bernoulli = model(bx)
        predict = torch.max(scores, dim = 1)
        micro_p, micro_r, micro_f, macro_p, macro_r, macro_f = \
            mf(preds=predict, golds=by, label=[i for i in range(1,args.class_num)])
        say("micro_p=%.4f, micro_r=%.4f, micro_f=%.4f, macro_p=%.4f, macro_r=%.4f, macro_f=%.4f\n"
            % (micro_p, micro_r, micro_f, macro_p, macro_r, macro_f))

def mf(self, preds, golds, label):
    # corrects = float(sum(preds == golds))
    # label = [i for i in range(1,51)]
    accuracy=metrics.accuracy_score(golds, preds)
    micro_p=metrics.precision_score(golds, preds,labels=label,  average="micro")
    micro_r=metrics.recall_score(golds, preds,labels=label, average="micro")
    micro_f=metrics.f1_score(golds, preds,labels=label, average="micro")
    macro_p=metrics.precision_score(golds, preds,labels=label, average="macro")
    macro_r=metrics.recall_score(golds, preds,labels=label, average="macro")
    macro_f=metrics.f1_score(golds, preds,labels=label, average="macro")
    return micro_p, micro_r, micro_f, macro_p, macro_r, macro_f





def main(args):
    print(args)
    assert args.embedding, "Pre-trained word embeddings required."
    embedding_loader = myio.embedding_loader(args.embedding)
    if args.train:
        train_x, train_y = myio.read_annotations(args.train)
        train_x = [ embedding_loader.map_words_to_indexes(x)[:args.max_len] for x in train_x ]

    if args.valid:
        valid_x, valid_y = myio.read_annotations(args.dev)
        valid_x = [ embedding_loader.map_words_to_indexes(x)[:args.max_len] for x in valid_x ]

    if args.test_json:
        test_data = myio.read_rationales(args.test_json)
        for x in test_data:
            x["x"] = filter(lambda xi: xi != "<padding>", x["x"])[:args.max_len]
            x["xids"] = embedding_loader.map_words_to_indexes(x["x"])
    if args.train:
        model = Model(
                    args = args,
                    embedding_loader = embedding_loader
                )
        if args.use_gpu and torch.cuda.is_available():
            model = model.cuda()
        train(args, model,
              train_data=(train_x,train_y),
              valid_data=(valid_x,valid_y),
              test_data=test_data)



if __name__=="__main__":
    args = options.load_arguments()
    main(args)