# -*- coding: utf-8 -*-

import sys
import argparse
import time
import numpy as np
import matplotlib as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

from da import DA
import load_data


class SDA:
    def __init__(
        self,
        rng,
        data,
        target,
        n_inputs=9600,
        n_hidden=[9600,9600,9600],
        n_outputs=2,
        corruption_levels=[0.1,0.2,0.3],
        gpu=-1):

        self.model = FunctionSet(l1=F.Linear(n_inputs, n_hidden[0]),
                                 l2=F.Linear(n_hidden[0], n_hidden[1]),
                                 l3=F.Linear(n_hidden[1], n_hidden[2]),
                                 l4=F.Linear(n_hidden[2], n_outputs))

        if gpu >= 0:
            self.model.to_gpu()
            self.xp = cuda.cupy
        else:
            self.xp = np

        self.rng = rng
        self.gpu = gpu
        self.data = data
        self.target = target

        self.x_train, self.x_test = data
        self.y_train, self.y_test = target

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        self.corruption_levels = corruption_levels
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.dae1 = None
        self.dae2 = None
        self.dae3 = None
        self.optimizer = None
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self.model)

    def pre_train(self, n_epoch=20, batchsize=100):
        first_inputs = self.data

        # initialize first dAE
        self.dae1 = DA(self.rng,
                       data=first_inputs,
                       n_inputs=self.n_inputs,
                       n_hidden=self.n_hidden[0],
                       corruption_level=self.corruption_levels[0],
                       gpu=self.gpu)
        # train first dAE
        print("--------First DA training has started!--------")
        self.dae1.train_and_test(n_epoch=n_epoch, batchsize=batchsize)
        self.dae1.to_cpu()
        # compute second iputs for second dAE
        tmp1 = self.dae1.compute_hidden(first_inputs[0])
        tmp2 = self.dae1.compute_hidden(first_inputs[1])
        if self.gpu >= 0:
            self.dae1.to_gpu()
        second_inputs = [tmp1, tmp2]


        # initialize second dAE
        self.dae2 = DA(self.rng,
                       data=second_inputs,
                       n_inputs=self.n_hidden[0],
                       n_hidden=self.n_hidden[1],
                       corruption_level=self.corruption_levels[1],
                       gpu=self.gpu)
        # train second dAE
        print("--------Second DA training has started!--------")
        self.dae2.train_and_test(n_epoch=n_epoch, batchsize=batchsize)
        self.dae2.to_cpu()
        # compute third inputs for third dAE
        tmp1 = self.dae2.compute_hidden(second_inputs[0])
        tmp2 = self.dae2.compute_hidden(second_inputs[1])
        if self.gpu >= 0:
            self.dae2.to_gpu()
        third_inputs = [tmp1, tmp2]



        # initialize third dAE
        self.dae3 = DA(self.rng,
                       data=third_inputs,
                       n_inputs=self.n_hidden[1],
                       n_hidden=self.n_hidden[2],
                       corruption_level=self.corruption_levels[2],
                       gpu=self.gpu)
        # train third dAE
        print("--------Third DA training has started!--------")
        self.dae3.train_and_test(n_epoch=n_epoch, batchsize=batchsize)

        # update model parameters
        self.model.l1 = self.dae1.encoder()
        self.model.l2 = self.dae2.encoder()
        self.model.l3 = self.dae3.encoder()

        self.setup_optimizer()

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.dropout(F.relu(self.model.l1(x)), train=train)
        h = F.dropout(F.relu(self.model.l2(h)), train=train)
        h = F.dropout(F.relu(self.model.l3(h)), train=train)
        y = self.model.l4(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fine_tune(self, n_epoch=20, batchsize=100):
        train_accs = []
        test_accs = []

        for epoch in range(1, n_epoch+1):
            print('fine tuning epoch ', epoch)

            perm = self.rng.permutation(self.n_train)
            sum_accuracy = 0
            sum_loss = 0
            for i in range(0, self.n_train, batchsize):
                x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
                y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.forward(x_batch, y_batch)
                loss.backward()
                self.optimizer.update()

                sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print('fine tuning train mean loss={}, accuracy={}'.format(sum_loss/self.n_train, sum_accuracy/self.n_train))
            train_accs.append(sum_accuracy/self.n_train)

            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for i in range(0, self.n_test, batchsize):
                x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
                y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

                real_batchsize = len(x_batch)

                loss, acc = self.forward(x_batch, y_batch, train=False)

                sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print('fine tuning test mean loss={}, accuracy={}'.format(sum_loss/self.n_test, sum_accuracy/self.n_test))
            test_accs.append(sum_accuracy/self.n_test)
        return train_accs, test_accs

if __name__ == '__main__':
    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=-1,            help='1: use gpu, 0: use cpu')
    parser.add_argument('--data '    , dest='data'       , type=str, default='input.dat',  help='an input data file')
    parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=40,           help='learning minibatch size')
    parser.add_argument('--nunits'   , dest='nunits'     , type=int, default=200,          help='number of units')

    args = parser.parse_args()
    batchsize   = args.batchsize    # minibatch size
    n_epoch     = args.epoch        # エポック数(パラメータ更新回数)

    #   Prepare dataset
    dataset, height, width = load_data.load_data(args.data)
    dataset['source'] = dataset['source'].astype(np.float32) #特徴量
    dataset['target'] = dataset['target'].astype(np.int32) #ラベル

    data_train,\
    data_test,\
    target_train,\
    target_test = train_test_split(dataset['source'], dataset['target'])

    data = [data_train, data_test]
    target = [target_train, target_test]

    rng = np.random.RandomState(1)


    start_time = time.time()

    sda = SDA(rng=rng,
              data=data,
              target=target,
              gpu=args.gpu)
    sda.pre_train(n_epoch=15)
    sda.fine_tune(n_epoch=20)

    end_time = time.time()

    print("time = {} min".format((end_time-start_time)/60.0))
