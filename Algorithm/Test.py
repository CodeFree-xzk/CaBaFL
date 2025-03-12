import copy
import json
from typing import List
from models import test_img
from models.Update import LocalUpdate_KD, LocalUpdate_FedAvg


def test(net_glob, dataset_test, args, idx=None):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args, idx)
    return acc_test.item()


def exp(model, dict_users, dataset_test, dataset_train, args):
    model_2 = copy.deepcopy(model)
    model_1 = copy.deepcopy(model)
    nonIID_dict: List[List[int]] = [[] for _ in range(10)]
    dict_test: List[List[int]] = [[] for _ in range(10)]
    for i in range(50000):
        nonIID_dict[dataset_train[i][1]].append(i)
    for i in range(10000):
        dict_test[dataset_test[i][1]].append(i)
    with open("./data/cifar10_10_iid_beta0.1.json", "r", encoding="utf-8") as f:
        data_dicts = json.load(f)
        iid_dict = data_dicts["train_data"]

    nonIID_dict = dict_users

    for i in range(args.num_users):
        print("-" * 100)
        local = LocalUpdate_KD(args, model, dataset_train, nonIID_dict[i])
        local.train(1, model, 1)
        print("KD", test(model, dataset_test, args))
        for j in range(10):
            print("{:.2f}".format(test(model, dataset_test, args, dict_test[j])), end="\t")
        print()

        local = LocalUpdate_FedAvg(args, dataset_train, nonIID_dict[i])
        local.train(0, model_2)
        print("noKD", test(model_2, dataset_test, args))
        for j in range(10):
            print("{:.2f}".format(test(model_2, dataset_test, args, dict_test[j])), end="\t")
        print()

        local = LocalUpdate_FedAvg(args, dataset_train, iid_dict[str(i)])
        local.train(0, model_1)
        print("IID", test(model_1, dataset_test, args))
        for j in range(10):
            print("{:.2f}".format(test(model_1, dataset_test, args, dict_test[j])), end="\t")
        print()
