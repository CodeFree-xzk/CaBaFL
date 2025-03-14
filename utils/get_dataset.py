#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets, transforms

from utils import mydata
from utils.ShakeSpare import ShakeSpeare
from utils.sampling import *
from utils.dataset_utils import separate_data, read_record
from utils.FEMNIST import FEMNIST
from torch.autograd import Variable
import torch.nn.functional as F
import os
import json


def get_dataset(args):
    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    file += ".json"
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.generate_data:
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar10':
        if args.data_augmentation == 1:
            trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                           std=[0.247, 0.243, 0.262])])
        else:
            trans_cifar10_train = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                           std=[0.247, 0.243, 0.262])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar100':
        if args.data_augmentation == 1:
            trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                            std=[0.267, 0.256, 0.276])])
        else:
            trans_cifar100_train = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                            std=[0.267, 0.256, 0.276])])
        trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                      std=[0.267, 0.256, 0.276])])
        if args.cifar100_coarse == 0:
            dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True,
                                              transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True,
                                             transform=trans_cifar100_val)
        else:
            dataset_train = mydata.CIFAR100_coarse('./data/cifar100', train=True, download=True,
                                                   transform=trans_cifar100_train)
            dataset_test = mydata.CIFAR100_coarse('./data/cifar100', train=False, download=True,
                                                  transform=trans_cifar100_val)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans)
        if args.generate_data:
            if args.iid:
                dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(True)
        dataset_test = FEMNIST(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
    elif args.dataset == 'ShakeSpare':
        dataset_train = ShakeSpeare(True)
        dataset_test = ShakeSpeare(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
    else:
        exit('Error: unrecognized dataset')

    if args.generate_data:
        with open(file, 'w') as f:
            dataJson = {"dataset": args.dataset, "num_users": args.num_users, "iid": args.iid,
                        "noniid_case": args.noniid_case, "data_beta": args.data_beta, "train_data": dict_users}
            json.dump(dataJson, f)

    return dataset_train, dataset_test, dict_users
