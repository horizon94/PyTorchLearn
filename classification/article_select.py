import os
import torch
import copy
from torch.utils.data import DataLoader
# import utils.DataProcessing as DP
import classification.models.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import classification.myio_article as myio
import time
import utils.utils.say as say
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
        valid_batches_x, valid_batches_a = myio.create_batches(
            valid_data[0], valid_data[1], args.batch_size, padding_id, args.max_len
        )
    if test_data is not None:
        test_batches_x, test_batches_a, test_batches_num = myio.create_batches_with_num(
            [i["xids"] for i in test_data],
            # [i["y"] for i in test_data],
            [i["articles"] for i in test_data],
            [i["number"] for i in test_data],
            args.batch_size,
            padding_id,
            args.max_len,
            sort=False
        )
    best_valid_loss = 1e100
    for epoch in range(args.max_epoches):
        train_batches_x, train_batches_a = myio.create_batches(
            train_data[0], train_data[1], args.batch_size, padding_id, args.max_len
        )
        epoch_start_time = time.time()
        train_epoch_loss = 0.
        batch_num = len(train_batches_x)
        for b in range(batch_num):
            # bx = torch.FloatTensor(train_batches_x[b])
            # by = torch.LongTensor(train_batches_y[b])
            bx = train_batches_x[b]
            ba = train_batches_a[b]
            if args.use_gpu and torch.cuda.is_available():
                bx, ba = Variable(bx.cuda()), Variable(ba.cuda())
            else:
                bx, ba = Variable(bx), Variable(ba)
            model.zero_grad()
            scores = model(bx)
            # print(scores.type)
            # print(by.type)
            # if args.loss_func is "mse":
            loss_func = nn.MSELoss()
            activ_func = nn.Sigmoid()
            loss = loss_func(activ_func(scores), ba)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.data[0]
        epoch_train_time = time.time() - epoch_start_time
        valid_loss = valid(args, model, valid_batches_x, valid_batches_a)
        say('epoch %s train loss %.3f traintime %s validate loss %.3f\n' % (
            epoch, train_epoch_loss / batch_num, int(epoch_train_time), valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            say("BETTER MODEL! SAVING...")
            model.save_model(args.save_model)
            say("\n")
            # test(args, model, test_batches_x, test_batches_y, test_batches_num)

def valid(args, model,valid_batches_x, valid_batches_a):
    valid_batch_num = len(valid_batches_x)
    valid_total_loss = 0.
    for b in range(valid_batch_num):
        print('.', end='', flush=True)
        # bx = torch.FloatTensor(valid_batches_x[b])
        # by = torch.LongTensor(valid_batches_y[b])
        bx = valid_batches_x[b]
        ba = valid_batches_a[b]
        if args.use_gpu and torch.cuda.is_available():
            bx, ba = Variable(bx.cuda()), Variable(ba.cuda())
        else:
            bx, ba = Variable(bx), Variable(ba)
        scores, pz, z, rationales, z_sizes, z_bernoulli = model(bx)
        loss_func = nn.MSELoss()
        activ_func = nn.Sigmoid()
        loss = loss_func(activ_func(scores), ba)
        valid_total_loss += loss.data[0]
    return valid_total_loss / valid_batch_num
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
              train_data=(train_x,train_a),
              valid_data=(valid_x,valid_a),
              test_data=test_data)

if __name__=='__main__':
    args = options.load_arguments()
    main(args)
    ### parameter setting
    # embedding_dim = 100
    # hidden_dim = 50
    # sentence_len = 32
    # train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    # test_file = os.path.join(DATA_DIR, TEST_FILE)
    # fp_train = open(train_file, 'r')
    # train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    # filenames = copy.deepcopy(train_filenames)
    # fp_train.close()
    # fp_test = open(test_file, 'r')
    # test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    # fp_test.close()
    # filenames.extend(test_filenames)
    #
    # corpus = DP.Corpus(DATA_DIR, filenames)
    # nlabel = 8
    #
    # ### create model
    # model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
    #                        vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    # if use_gpu:
    #     model = model.cuda()
    # ### data processing
    # dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
    #
    # train_loader = DataLoader(dtrain_set,
    #                       batch_size=batch_size,
    #                       shuffle=True,
    #                       num_workers=4
    #                      )
    # dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)
    #
    # test_loader = DataLoader(dtest_set,
    #                       batch_size=batch_size,
    #                       shuffle=False,
    #                       num_workers=4
    #                      )
    # for idx, (name,_) in enumerate(model.named_parameters()):
    #     print(idx, name)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # loss_function = nn.CrossEntropyLoss()
    # train_loss_ = []
    # test_loss_ = []
    # train_acc_ = []
    # test_acc_ = []
    # ### training procedure
    # for epoch in range(epochs):
    #     optimizer = adjust_learning_rate(optimizer, epoch)
    #
    #     ## training epoch
    #     total_acc = 0.0
    #     total_loss = 0.0
    #     total = 0.0
    #     for iter, traindata in enumerate(train_loader):
    #         train_inputs, train_labels = traindata
    #         train_labels = torch.squeeze(train_labels)
    #
    #         if use_gpu:
    #             train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
    #         else: train_inputs = Variable(train_inputs)
    #
    #         model.zero_grad()
    #         model.batch_size = len(train_labels)
    #         model.hidden = model.init_hidden()
    #         print("train_inputs",train_inputs.shape)
    #         output = model(train_inputs.t())
    #         print(output.data[0])
    #         loss = loss_function(output, Variable(train_labels))
    #         loss.backward()
    #         optimizer.step()
    #
    #         # calc training acc
    #         _, predicted = torch.max(output.data, 1)
    #         total_acc += (predicted == train_labels).sum()
    #         total += len(train_labels)
    #         total_loss += loss.data[0]
    #
    #     train_loss_.append(total_loss / total)
    #     train_acc_.append(total_acc / total)
    #     ## testing epoch
    #     total_acc = 0.0
    #     total_loss = 0.0
    #     total = 0.0
    #     for iter, testdata in enumerate(test_loader):
    #         test_inputs, test_labels = testdata
    #         test_labels = torch.squeeze(test_labels)
    #
    #         if use_gpu:
    #             test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
    #         else: test_inputs = Variable(test_inputs)
    #
    #         model.batch_size = len(test_labels)
    #         model.hidden = model.init_hidden()
    #         output = model(test_inputs.t())
    #
    #         loss = loss_function(output, Variable(test_labels))
    #
    #         # calc testing acc
    #         _, predicted = torch.max(output.data, 1)
    #         total_acc += (predicted == test_labels).sum()
    #         total += len(test_labels)
    #         total_loss += loss.data[0]
    #     test_loss_.append(total_loss / total)
    #     test_acc_.append(total_acc / total)
    #
    #     print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
    #           % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))
    #
    # param = {}
    # param['lr'] = learning_rate
    # param['batch size'] = batch_size
    # param['embedding dim'] = embedding_dim
    # param['hidden dim'] = hidden_dim
    # param['sentence len'] = sentence_len
    #
    # result = {}
    # result['train loss'] = train_loss_
    # result['test loss'] = test_loss_
    # result['train acc'] = train_acc_
    # result['test acc'] = test_acc_
    # result['param'] = param
    #
    # if use_plot:
    #     import PlotFigure as PF
    #     PF.PlotFigure(result, use_save)
    # if use_save:
    #     filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
    #     result['filename'] = filename
    #
    #     fp = open(filename, 'wb')
    #     pickle.dump(result, fp)
    #     fp.close()
    #     print('File %s is saved.' % filename)
