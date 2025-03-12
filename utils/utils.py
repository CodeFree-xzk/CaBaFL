#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os
import time


def save_result(data, ylabel, args):
    data = {'base': data}

    if args.iid == 0:
        path = './output/{}'.format(args.noniid_case)
    else:
        path = './output/iid'.format(args.noniid_case)

    if args.noniid_case == 5:
        path += '/{}'.format(args.data_beta)

    file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                ylabel, args.epochs, args.lr,
                                                datetime.datetime.now().strftime(
                                                    "%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

def save_result_1(args, acc, time_list):
    if args.iid == 0:
        path = './output/{}'.format(args.noniid_case)
        if args.noniid_case == 5:
            path += '/{}'.format(args.data_beta)
    else:
        path = './output/iid'.format(args.noniid_case)

    file = '{}_{}_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                       datetime.datetime.now().strftime("%m_%d_%H_%M_%S"),)

    if not os.path.exists(path):
        os.makedirs(path)
    acc = [str(i) for i in acc]
    time_list = [str(i) for i in time_list]
    with open(os.path.join(path, file), 'a') as f:
        f.write(" ".join(acc))
        f.write('\n')
        f.write(" ".join(time_list))
        f.write('\n')
    print('save finished')
    f.close()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
