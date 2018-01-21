import os
import torch
import copy
from torch.utils.data import DataLoader
# import utils.DataProcessing as DP
import sys
sys.path.append("../")
from classification.models.LSTMClassifier import LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import classification.myio_article as myio
import time
from utils.utils import  say
import classification.options_article as options
use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

## parameter setting
epochs = 50
batch_size = 5
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(args, model,train_data, valid_data, test_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    padding_id = model.embedding_loader.map_word_to_index("<padding>")
    if valid_data is not None:
        valid_batches_x, valid_batches_y,valid_batches_a = myio.create_batches(
            valid_data[0], valid_data[1], valid_data[2], args.batch_size, padding_id, args.max_len
        )
    if test_data is not None:
        test_batches_x, test_batches_y, test_batches_a, test_batches_num = myio.create_batches_with_num(
            [i["xids"] for i in test_data],
            [i["y"] for i in test_data],
            [i["articles"] for i in test_data],
            [i["number"] for i in test_data],
            args.batch_size,
            padding_id,
            args.max_len,
            sort=False
        )
    best_valid_loss = 1e100
    for epoch in range(args.max_epoches):
        train_batches_x, train_batches_y, train_batches_a = myio.create_batches(
            train_data[0], train_data[1],train_data[2], args.batch_size, padding_id, args.max_len
        )
        epoch_start_time = time.time()
        train_epoch_loss = 0.
        batch_num = len(train_batches_x)
        for b in range(batch_num):
            # bx = torch.FloatTensor(train_batches_x[b])
            # by = torch.LongTensor(train_batches_y[b])
            bx = train_batches_x[b]
            by = train_batches_y[b]
            ba = train_batches_a[b]
            if args.use_gpu and torch.cuda.is_available():
                bx, by, ba = Variable(bx.cuda()), Variable(by.cuda()), Variable(ba.cuda())
            else:
                bx, by, ba = Variable(bx), Variable(by), Variable(ba)
            model.zero_grad()
            scores_class, scores_article = model(bx)
            # print(scores.type)
            # print(by.type)
            # if args.loss_func is "mse":
            class_loss_func = nn.MSELoss()
            class_activ_func = nn.Sigmoid()
            article_loss_func = nn.MSELoss()
            article_activ_func = nn.Sigmoid()
            class_loss = class_loss_func(class_activ_func(scores_class), by)
            article_loss = article_loss_func(article_activ_func(scores_class), ba)
            loss = class_loss + article_loss
            loss.backward()
            optimizer.step()
            if (b+1) % 100 == 0:
                print("train_batch_class_loss=",class_loss.data[0],"train_batch_article_loss=",article_loss.data[0])
            train_epoch_loss += loss.data[0]
        epoch_train_time = time.time() - epoch_start_time
        valid_loss = valid(args, model, valid_batches_x, valid_batches_a)
        say('epoch %s train loss %.3f traintime %s validate loss %.3f\n' % (
            epoch, train_epoch_loss / batch_num, int(epoch_train_time), valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            say("BETTER MODEL! SAVING...")
            #model.save_model(args.save_model)
            #say("\n")
            p_1, p_3, p_5, p_10 = test(args, model, test_batches_x, test_batches_a)
            say("p@1=%.3f, p@3=%.3f, p@5=%.3f, p@10=%.3f" %(p_1, p_3, p_5, p_10))

def valid(args, model,valid_batches_x, valid_batches_a):
    valid_batch_num = len(valid_batches_x)
    valid_total_loss = 0.
    for b in range(valid_batch_num):
        #print('.', end='', flush=True)
        # bx = torch.FloatTensor(valid_batches_x[b])
        # by = torch.LongTensor(valid_batches_y[b])
        bx = valid_batches_x[b]
        ba = valid_batches_a[b]
        if args.use_gpu and torch.cuda.is_available():
            bx, ba = Variable(bx.cuda()), Variable(ba.cuda())
        else:
            bx, ba = Variable(bx), Variable(ba)
        scores = model(bx)
        loss_func = nn.MSELoss()
        activ_func = nn.Sigmoid()
        loss = loss_func(activ_func(scores), ba)
        valid_total_loss += loss.data[0]
    return valid_total_loss / valid_batch_num
def test(args, model, test_batches_x, test_batches_a):
    test_batch_num = len(test_batches_x)
    scores_total=None
    golds_total=None
    for b in range(test_batch_num):
        bx = test_batches_x[b]
        ba = test_batches_a[b]
        if args.use_gpu and torch.cuda.is_available():
             bx, ba = Variable(bx.cuda()), Variable(ba.cuda())
        else:
             bx, ba = Variable(bx), Variable(ba)
        scores = model(bx)
        if scores_total is not None:
            golds_total = torch.cat((golds_total,ba), dim = 0)
            scores_total = torch.cat((scores_total,scores), dim = 0)
        else:
            golds_total = ba
            scores_total = scores
    return MF_DOC(scores_total, golds_total)
    
    





def MF_DOC(scores, golds):
    '''
         calculate p@1 p@3 p@5 p@10
    '''
    result=[]
    batch_size = scores.shape[0]
    #scores shape(batch_size,article_num)
    sorted_tensor, indices = torch.sort(scores, dim=1, descending=True)
    #indices shape(batch_size, article_num)
    for i in [1,3,5,10]:
        right = 0
        choosed=indices[:,i]  # shape(batch_size, i)
        for b in range(batch_size):
            choose_one_doc = choosed[b].data[:]
            gold_one_doc = golds[b].data[:]
            for doc in choose_one_doc:
                if gold_one_doc[doc]>0.:
                    right += 1
        p=right/(i*batch_size)
        result.append(p)
    return result
def main(args):
    print(args)
    # assert args.embedding, "Pre-trained word embeddings required."
    embedding_loader = myio.embedding_loader(args.embedding,
                                             args.embedding_dim
                                             )
    if args.train:
        train_x, train_y, train_a = myio.read_annotations(args.train)
        train_x = [embedding_loader.map_words_to_indexes(x)[:args.max_len] for x in train_x]

    if args.valid:
        valid_x, valid_y, valid_a = myio.read_annotations(args.valid)
        valid_x = [embedding_loader.map_words_to_indexes(x)[:args.max_len] for x in valid_x]

    if args.test_json:
        test_data = myio.read_rationales(args.test_json)
        for x in test_data:
            x["x"] = list(filter(lambda xi: xi != "<padding>", x["x"]))[:args.max_len]
            x["xids"] = embedding_loader.map_words_to_indexes(x["x"])
    if args.train:
        model = LSTMC(
                    args = args,
                    embedding_loader = embedding_loader
                )
        if args.use_gpu and torch.cuda.is_available():
            model = model.cuda()
        train(args, model,
              train_data=(train_x,train_y,train_a),
              valid_data=(valid_x,valid_y,valid_a),
              test_data=test_data)

if __name__=='__main__':
    args = options.load_arguments()
    main(args)
