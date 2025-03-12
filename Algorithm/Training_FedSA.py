import copy

import numpy as np
import wandb

from models.Fed import *
from models.Update import LocalUpdate_FedSA
from utils.Clients import Clients
from utils.utils import *
from models.test import test_img


def FedSA(args, net_glob, dataset_train, dataset_test, dict_users):
    group = "IID"
    if args.iid == 0:
        group = str(args.data_beta)
    wandb.init(project="FedTrace", name=group + "_" + args.algorithm, group=group,
               tags=[str(args.model), args.dataset])
    wandb.log({'acc': 0, 'max_avg': 0, 'time': 0})
    net_glob.train()
    acc = []
    max_avg = 0
    max_std = 0
    time_list = []
    comm_list = []
    comm_count = 0

    select_list = [0 for _ in range(args.num_users)]
    clients = Clients(args, dict_users)
    start_time = 0

    m = int(args.num_users * args.frac)
    M = int(args.M_frac * m)

    lens = []
    for idx in range(args.num_users):
        lens.append(len(dict_users[idx]))

    for iter in range(1, args.epochs + 1):
        if start_time > args.limit_time:
            break
        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        if iter == 1:
            train = clients.get_idle(m)
        else:
            train = clients.get_idle(M)
        for idx in train:
            clients.train(idx, iter - 1, copy.deepcopy(net_glob))
            comm_count += 0.5
        for idx, version, model, time in clients.update_list:
            client = clients.get(idx)
            if client.version < iter - args.max_tolerate:
                clients.train(idx, iter - 1, copy.deepcopy(net_glob))
                comm_count += 0.5

        for idx in train:
            select_list[idx] += 1

        lens = {}
        for idx, version, model, time in clients.update_list:
            lens[idx] = len(dict_users[idx])
        update_users = clients.pop_update(M)
        update_w = {}
        for idx, version, model, time in update_users:
            local = LocalUpdate_FedSA(args=args, dataset=dataset_train, idxs=dict_users[idx])
            lr = args.lr / (args.num_users * (select_list[idx] / sum(select_list)))
            w = local.train(net=model, lr=lr)
            update_w[idx] = w

        comm_count += 0.5 * M
        start_time += update_users[-1][-1]

        w_glob = copy.deepcopy(net_glob).state_dict()
        w_glob = Weighted_Aggregation_FedSA(update_w, lens, w_glob)

        net_glob.load_state_dict(w_glob)

        for idx, version, model, time in update_users:
            clients.get(idx).version = iter

        # M = estimate_M()

        acc_value = test(net_glob, dataset_test, args)
        acc.append(acc_value)
        time_list.append(start_time)
        if len(acc) >= 10:
            avg = sum(acc[len(acc) - 10::]) / 10
            if avg > max_avg:
                max_avg = avg
                max_std = np.std(acc[len(acc) - 10::])
        wandb.log({'acc': acc_value, 'max_avg': max_avg, 'time': start_time, 'comm_times': comm_count})
        time_list.append(start_time)
        comm_list.append(comm_count)
        print(max_std)

    save_result(acc, 'test_acc', args)
    save_result(time_list, 'test_time', args)
    save_result(comm_list, 'test_comm', args)


def estimate_M():
    pass


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()
