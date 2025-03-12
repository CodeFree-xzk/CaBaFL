#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import wandb
from Algorithm.Training_FedSA import FedSA
from Algorithm.Training_CaBaFL import CaBaFL
from Algorithm.Training_WKAFL import WKAFL
from Algorithm.Test import exp
from models.Fed import Weighted_Aggregation_FedASync
from utils.Clients import Clients
from utils.options import args_parser
from models import *
from models.LSTM import *
from utils.get_dataset import get_dataset
from utils.utils import *
from utils.set_seed import set_random_seed
from Algorithm import *


def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    # training
    acc = []
    time_list = []
    clients = Clients(args, dict_users)
    start_time = 0
    if args.iid == 0:
        wandb.init(project="FedTrace", name=str(args.data_beta) + "_" + args.algorithm, group=str(args.data_beta),
                   tags=[str(args.model), args.dataset])
    else:
        wandb.init(project="FedTrace", name="IID_" + args.algorithm, group="IID", tags=[str(args.model), args.dataset])
    wandb.log({'acc': 0, 'max_avg': 0, 'time': 0})
    max_avg = 0
    max_std = 0
    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break
        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter, net_glob)
        # update global weights
        start_time += clients.pop_update(m)[-1][-1]
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc_value = test(net_glob, dataset_test, args)
        acc.append(acc_value)
        time_list.append(start_time)
        if len(acc) >= 10:
            avg = sum(acc[len(acc) - 10::]) / 10
            if avg > max_avg:
                max_avg = avg
                max_std = np.std(acc[len(acc) - 10::])
        wandb.log({'acc': acc_value, 'max_avg': max_avg, 'time': start_time})
        print(max_std)
    save_result_1(args, acc, time_list)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
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
    clients = Clients(args, dict_users)
    start_time = 0

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter, None)
        # update global weights
        start_time += clients.pop_update(m)[-1][-1]
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc_value = test(net_glob, dataset_test, args)
        acc.append(acc_value)
        time_list.append(start_time)
        if len(acc) >= 10:
            avg = sum(acc[len(acc) - 10::]) / 10
            if avg > max_avg:
                max_avg = avg
                max_std = np.std(acc[len(acc) - 10::])
        wandb.log({'acc': acc_value, 'max_avg': max_avg, 'time': start_time})
        print(max_std)
        time_list.append(start_time)

    save_result(acc, 'test_acc', args)
    save_result(time_list, 'test_time', args)


def FedGKD(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []
    time_list = []
    clients = Clients(args)
    start_time = 0

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedGKD(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter)
        # update global weights
        start_time += clients.pop_update(m)[-1][2]
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))
        time_list.append(start_time)

    save_result(acc, 'test_acc', args)
    save_result(time_list, 'test_time', args)


def Moon(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.to('cpu')
    net_glob.train()

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    old_nets_pool = [[] for i in range(args.num_users)]

    acc = []
    time_list = []
    clients = Clients(args)
    start_time = 0

    lens = [len(datasets) for _, datasets in dict_users.items()]

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)
        w_glob = {}
        total_len = 0
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_Moon(args=args, glob_model=net_glob, old_models=old_nets_pool[idx],
                                     dataset=dataset_train, idxs=dict_users[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)

            w_local = local.train(round=iter, net=net_local.to(args.device))

            clients.train(idx, iter)

            # update global weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key].to('cpu')
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key].to('cpu')

            total_len += len(dict_users[idx])

            if len(old_nets_pool[idx]) < args.model_buffer_size:
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                old_nets_pool[idx].append(old_net.to('cpu'))
            elif args.pool_option == 'FIFO':
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[idx][i] = old_nets_pool[idx][i + 1]
                old_nets_pool[idx][args.model_buffer_size - 1] = old_net.to('cpu')

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        start_time += clients.pop_update(m)[-1][2]
        acc.append(test(net_glob, dataset_test, args))
        time_list.append(start_time)

    save_result(acc, 'test_acc', args)
    save_result(time_list, 'test_time', args)


from utils.clustering import *
from scipy.cluster.hierarchy import linkage


def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []
    time_list = []
    clients = Clients(args)
    start_time = 0

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)
            clients.train(idx, iter)

            del local_model
        # update global weights
        start_time += clients.pop_update(len(idxs_users))[-1][2]
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        acc.append(test(net_glob, dataset_test, args))
        time_list.append(start_time)
        net_glob.to('cpu')

        del clients_models

    save_result(acc, 'test_acc', args)
    save_result(time_list, 'test_time', args)


def FedASync(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []
    acc_10 = []
    time_list = []
    comm_list = []
    update_list = []
    idle_set = set(list(range(args.num_users)))
    comm = 0
    max_avg = 0
    max_std = 0
    group = "IID"
    if args.iid == 0:
        group = str(args.data_beta)
    wandb.init(project="FedTrace", name=group + "_" + args.algorithm, group=group,
               tags=[str(args.model), args.dataset])
    wandb.log({'acc': 0, 'max_avg': 0, 'time': 0})

    clients = Clients(args, dict_users)
    start_time = 0
    m = max(int(args.frac * args.num_users), 1)
    init_users = np.random.choice(range(args.num_users), m, replace=False)

    for model_index, client_index in enumerate(init_users):
        update_list.append([client_index, copy.deepcopy(net_glob), 0, clients.getTime(client_index)])
        idle_set.remove(client_index)

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        update_list.sort(key=lambda x: x[-1])
        update_idx, model_copy, version, time = update_list.pop(0)
        for i in update_list:
            i[-1] -= time

        lag = iter - version
        start_time += time

        alpha = args.FedASync_alpha * ((lag + 1) ** -args.poly_a)
        print("lag:", lag)
        print("alpha:", alpha)

        local = LocalUpdate_FedASync(args=args, glob_model=model_copy, dataset=dataset_train,
                                     idxs=dict_users[update_idx])
        w = local.train(net=copy.deepcopy(net_glob).to(args.device))

        w_new = copy.deepcopy(net_glob.state_dict())
        w_new = Weighted_Aggregation_FedASync(w, w_new, alpha)
        net_glob.load_state_dict(w_new)

        acc_value = test(net_glob, dataset_test, args)
        acc.append(acc_value)
        time_list.append(start_time)
        comm += 2
        comm_list.append(comm)

        idx = np.random.choice(list(idle_set), 1, replace=False)[0]
        idle_set.remove(idx)
        idle_set.add(update_idx)
        update_list.append([idx, copy.deepcopy(net_glob), iter, clients.getTime(idx)])

        if iter % 10 == 0:
            acc_10.append(acc_value)
            if len(acc_10) >= 10:
                avg = sum(acc_10[len(acc_10) - 10::]) / 10
                if avg > max_avg:
                    max_avg = avg
                    max_std = np.std(acc_10[len(acc_10) - 10::])
            wandb.log({'acc': acc_value, 'max_avg': max_avg, 'time': start_time})
        print(max_std)
    save_result_1(args, acc, time_list)


def SAFA(args, net_glob, dataset_train, dataset_test, dict_users):
    group = "IID"
    if args.iid == 0:
        group = str(args.data_beta)
    wandb.init(project="FedTrace", name=group + "_" + args.algorithm, group=group,
               tags=[str(args.model), args.dataset])
    wandb.log({'acc': 0, 'max_avg': 0, 'time': 0})
    net_glob.to('cpu')
    net_glob.train()
    acc = []
    time_list = []
    comm_list = []
    comm_count = 0
    max_avg = 0
    max_std = 0

    clients = Clients(args, dict_users)
    start_time = 0

    cache = [copy.deepcopy(net_glob.state_dict()) for _ in range(args.num_users)]
    local_result = [0 for _ in range(args.num_users)]
    net_glob.to(args.device)

    pre_P = set()
    P = set()
    next_num = int(args.num_users * args.frac)

    for iter in range(1, args.epochs + 1):
        if start_time > args.limit_time:
            break
        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        outdated = set()
        train = clients.get_idle(next_num)
        for idx in train:
            clients.train(idx, iter - 1, copy.deepcopy(net_glob))
            comm_count += 0.5
        for idx, version, model, time in clients.update_list:
            client = clients.get(idx)
            if client.version < iter - args.max_tolerate:
                clients.train(idx, iter - 1, copy.deepcopy(net_glob))
                comm_count += 0.5

        update_users = clients.get_update_byLimit(args.limit)

        Q = []
        count = 0
        for idx, version, model, time in update_users:
            if len(P) == int(args.num_users * args.frac) * args.P_frac:
                break
            if idx not in pre_P:
                P.add(idx)
            else:
                Q.append(idx)
            count += 1
        update_users = update_users[0:count]

        for idx, version, model, time in update_users:
            local = LocalUpdate_SAFA(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=model)
            for key in w.keys():
                w[key] = w[key].cpu()
            local_result[idx] = w

        update_list = clients.update_list[::]
        clients.pop_update(count)
        comm_count += 0.5 * count
        start_time += min(args.limit, update_users[-1][-1])
        next_num = count

        if len(P) < args.num_users * args.frac:
            q = min(int(args.num_users * args.frac - len(P)), len(Q))
            P = P.union(set(Q[0:q]))
            Q = Q[q::]

        for idx in range(args.num_users):
            if idx in P:
                cache[idx] = local_result[idx]
            elif idx in outdated:
                cache[idx] = copy.deepcopy(net_glob.state_dict())

        c = []
        lens = []
        for idx, version, model, time in update_list:
            c.append(cache[idx])
            lens.append(len(dict_users[idx]))

        w_glob = Aggregation(c, lens)

        for idx in range(args.num_users):
            if idx in Q:
                cache[idx] = local_result[idx]

        net_glob.load_state_dict(w_glob)

        for idx, version, model, time in update_users:
            clients.get(idx).version = iter
        pre_P = P
        P = set()

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

    save_result_1(args, acc, time_list)


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


if __name__ == '__main__':    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.iid == 0 and args.data_beta == 0.1:
        args.time_c = 1
    if args.dataset == 'cifar100' and args.cifar100_coarse == 1:
        args.num_classes = 20
    elif args.dataset == 'cifar100' and args.cifar100_coarse == 0:
        args.num_classes = 100
    elif args.dataset == 'femnist':
        args.num_classes = 62
        args.num_channels = 1
        args.generate_data = 0
    elif args.dataset == 'ShakeSpare':
        args.num_classes = 80
        args.generate_data = 0
    # print(args.device)

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(len(dict_users))
    # print(dict_users)
    print([len(dict_users[i]) for i in dict_users])

    if args.dataset == 'femnist':
        net_glob = CNNFashionMnist(args)
    elif args.dataset == 'mnist':
        net_glob = CNNMnist(args)
    elif args.use_project_head:
        net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
    elif 'cnn' in args.model:
        net_glob = CNNCifar(args)
    elif 'resnet' in args.model:
        net_glob = ResNet18_cifar10(num_classes=args.num_classes, args=args)
    elif 'mobilenet' in args.model:
        net_glob = MobileNetV2(args)
    elif 'vgg' in args.model:
        net_glob = vgg16_bn(num_classes=args.num_classes)
    elif 'lstm' in args.model:
        net_glob = CharLSTM()

    net_glob.to(args.device)
    # print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'Scaffold':
        Scaffold(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'Moon':
        Moon(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGKD':
        FedGKD(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGen':
        FedGen(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedDC':
        FedDC(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedMLB':
        assert 'resnet' in args.model, 'Current, FedMLB only use resnet model!'
        FedMLB(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedASync':
        FedASync(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'SAFA':
        SAFA(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedSA':
        FedSA(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == "WKAFL":
        wkafl = WKAFL(args, net_glob, dataset_train, dataset_test, dict_users)
        wkafl.train()
    elif args.algorithm == "-":
        motivation(args, net_glob, dataset_train)
    elif args.algorithm == 'CaBaFL':
        cabafl = CaBaFL(args, net_glob, dataset_train, dataset_test, dict_users)
        cabafl.train_activation()
    else:
        exp(net_glob, dict_users, dataset_test, dataset_train, args)
