import copy
import math
import random
from models.Update import DatasetSplit
import torch
import numpy as np
from models import LocalUpdate_FedAvg, Aggregation, test_img
from utils.Clients import Clients
from utils.utils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


# 这是master
class FedTrace:
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, dataset_train_1):
        self.n = 0
        self.args = args
        self.client = Clients(args, dict_users)
        self.train_time_list = self.client.train_time
        self.net_glob = net_glob
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.global_version = 0
        self.cache_size = max(int(self.args.frac * self.args.num_users), 1)
        self.cache = [copy.deepcopy(net_glob.state_dict()) for _ in range(self.cache_size)]
        self.cache_update_times = [1 for _ in range(self.cache_size)]
        self.selected_count = [0 for _ in range(args.num_users)]
        self.true_label_count_lists = [np.zeros(args.num_classes) for _ in range(self.cache_size)]
        self.soft_label_count_lists = [[0 for _ in range(10)] for _ in range(self.cache_size)]

        self.cache_trace = [[] for _ in range(self.cache_size)]
        self.model_trace = [[] for _ in range(self.cache_size)]
        self.model_version_list = [0 for _ in range(self.cache_size)]
        self.true_labels = self.getTrueLabels()
        self.soft_labels = [[] for _ in range(args.num_users)]
        self.activation_labels = [torch.tensor([]) for _ in range(args.num_users)]
        self.update_activation()
        self.global_activation = torch.zeros(self.n).to(self.args.device)

        self.var_change_of_client = [[0, 0] for i in range(args.num_users)]
        self.sim_history = []
        self.ds_var_history = []
        self.model_version_diff_history = []
        self.var_list_cache = []
        self.var_list_end = []
        self.w_var_list = []

        self.time = 0
        self.acc = []
        self.time_list = []
        self.comm = 0
        self.comm_list = []

        self.idle_clients = set(list(range(args.num_users)))
        self.update_list = []
        self.select_var_count = 0
        self.label_var_count = 0
        self.max_avg = 0
        self.max_std = 0

        # self.dataset_train_1 = dataset_train_1
        # self.true_labels_1 = self.getTrueLabels_1()
        # self.true_label_count_lists_1 = [[0 for _ in range(100)] for _ in range(10)]
        # self.VAR = None
        #
        # for i in range(100):
        #     print(self.true_labels[i])
        #     print(self.true_labels_1[i])
        #     print(sum(self.true_labels_1[i]) == sum(self.true_labels[i]))

        # [client_index, model_copy, model_index,(model_version_1, model_version_2),global_model_version, train_time]
        if args.iid == 0:
            self.name = "{}_C{}_W{}_L{}".format(args.data_beta, args.C, args.W, args.label_type)
        else:
            self.name = "IID_C{}_W{}_L{}".format(args.C, args.W, args.label_type)
        if args.info != "":
            self.name += ("_" + args.info)

        if args.iid == 0:
            if args.data_beta == 0.1:
                self.select_var_bound = 3e-6
                self.a = 0.5
                self.threshold = 5000
            elif args.data_beta == 0.5:
                self.select_var_bound = 1e-6
                self.a = 0.7
                self.args.CB = 5
                self.threshold = 5000
                self.dynT = 15
            elif args.data_beta == 1.0:
                self.select_var_bound = 1e-6
                self.a = 0.9
                self.args.CB = 9
                self.threshold = 4000
                self.dynT = 16
        else:
            self.select_var_bound = 4e-6
            self.a = 0.9
            self.args.CB = 100
            self.threshold = 1000
            self.dynT = 20
        if args.simulate == 0:
            group = "IID"
            if args.iid != 1:
                group = str(args.data_beta)
            wandb.init(project=args.algorithm, name=self.name, group=group,
                       tags=[str(args.model), str(args.dataset)],
                       config={"CL": args.CL, "seed": args.seed,
                               "time": args.time_c, "CB": args.CB, "SM": args.SM})
            wandb.log({'acc': 0, 'max_avg': 0, 'time': 0, "comm": 0})

    def getTrueLabels(self, normal=False, dataset_train=None, num_classes=None, dict_users=None):
        trueLabels = []
        dataset_train = self.dataset_train if dataset_train is None else dataset_train
        num_classes = self.args.num_classes if num_classes is None else num_classes
        dict_users = self.dict_users if dict_users is None else dict_users
        for i in range(self.args.num_users):
            label = [0 for _ in range(num_classes)]
            for data_idx in dict_users[i]:
                label[dataset_train[data_idx][1]] += 1
            if normal:
                label = unitization(np.array(label))
            trueLabels.append(np.array(label))
        return trueLabels

    def update_softLabels(self, var=None):
        if var is None:
            var = list(range(self.args.num_users))
        self.net_glob.eval()
        with torch.no_grad():
            for idx, client_idx in enumerate(var):
                soft_label = torch.zeros(1, 10).to(self.args.device)
                # soft_label = torch.zeros(1, args.num_classes).to(args.device)
                ldr_train = DataLoader(DatasetSplit(self.dataset_train, self.dict_users[client_idx]),
                                       batch_size=self.args.local_bs, shuffle=True)
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images = images.to(self.args.device)
                    log_probs = F.softmax(self.net_glob(images)['output'], dim=1)
                    for probs in log_probs:
                        soft_label += probs

                # soft_label /= len(self.dict_users[client_idx])
                # print(soft_label)
                # print()
                self.soft_labels[client_idx] = soft_label
        for i in range(len(self.soft_labels)):
            self.soft_labels[i] = self.soft_labels[i][0].cpu().numpy()

    def update_soft_label_count_lists(self):
        for model_index, trace in enumerate(self.model_trace):
            temp_list = [0 for _ in range(10)]
            for client_index in trace:
                for i in range(self.args.num_classes):
                    temp_list[i] += self.soft_labels[client_index][i]
            self.soft_label_count_lists[model_index] = temp_list

    def update_cache_sample_list(self):
        for model_index, trace in enumerate(self.cache_trace):
            temp_list = [0 for _ in range(10)]
            for client_index in trace:
                for i in range(self.args.num_classes):
                    temp_list[i] += self.soft_labels[client_index][i]
            self.cache_sample_list[model_index] = temp_list

    def update_activation(self, var=None):
        if var is None:
            var = list(range(self.args.num_users))
        self.net_glob.eval()
        with torch.no_grad():
            for client_idx in var:
                activation_label = None
                # activation_label = torch.zeros(1, self.n).to(self.args.device)
                ldr_train = DataLoader(DatasetSplit(self.dataset_train, self.dict_users[client_idx]),
                                       batch_size=self.args.local_bs, shuffle=True)
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images = images.to(self.args.device)
                    log_probs = self.net_glob(images)['representation']
                    if self.args.SM == 1:
                        log_probs = F.softmax(log_probs, dim=1)
                    for probs in log_probs:
                        if activation_label is None:
                            activation_label = probs
                            self.n = len(activation_label)
                            continue
                        activation_label += probs

                self.activation_labels[client_idx] = activation_label
        global_activation = torch.zeros(self.n).to(self.args.device)
        for activation in self.activation_labels:
            global_activation += activation
        g = global_activation.cpu().numpy()
        self.global_activation = g
        for i in range(len(self.activation_labels)):
            self.activation_labels[i] = self.activation_labels[i].cpu().numpy()

    def test(self):
        acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
        self.acc.append(acc_test.item())
        if len(self.acc) >= 10:
            avg = sum(self.acc[len(self.acc) - 10::]) / 10
            if avg > self.max_avg:
                self.max_avg = avg
                self.max_std = np.std(self.acc[len(self.acc) - 10::])
        print("acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}".format(acc_test, self.max_avg, self.max_std))
        self.time_list.append(self.time)
        self.comm_list.append(self.comm)
        # wandb.log({'acc': acc_test.item(), 'max_avg': self.max_avg, 'time': self.time, "VAR": self.VAR, "sim_mean": np.mean(self.var_history)})
        wandb.log({'acc': acc_test.item(), 'max_avg': self.max_avg, 'time': self.time, "comm": self.comm})

        return acc_test.item()

    def save_result(self):
        if self.args.iid == 0:
            path = './output/{}'.format(self.args.noniid_case)
            if self.args.noniid_case == 5:
                path += '/{}'.format(self.args.data_beta)
        else:
            path = './output/iid'.format(self.args.noniid_case)

        file = '{}_{}_{}_{}_{}.txt'.format(self.args.dataset, self.args.algorithm, self.args.model,
                                           datetime.datetime.now().strftime("%m_%d_%H_%M_%S"),
                                           self.name)

        if not os.path.exists(path):
            os.makedirs(path)
        self.acc = [str(i) for i in self.acc]
        self.time_list = [str(i) for i in self.time_list]
        with open(os.path.join(path, file), 'a') as f:
            f.write(" ".join(self.acc))
            f.write('\n')
            f.write(" ".join(self.time_list))
            f.write('\n')
        print('save finished')
        f.close()

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    def get_cos_sim(self, vector1, vector2=None, normal=True):
        if vector2 is None:
            vector2 = self.global_activation
        if normal:
            vector1 = vector1 / np.linalg.norm(vector1)
            vector2 = vector2 / np.linalg.norm(vector2)
        dot = np.dot(vector1, vector2)
        return dot / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def get_acc_act(self, clients):
        activation = np.zeros(self.n)
        for clientIndex in clients:
            activation += self.activation_labels[clientIndex]
        return activation

    def get_acc_label(self, clients):
        acc_label = np.zeros(self.args.num_classes)
        for clientIndex in clients:
            acc_label += self.true_labels[clientIndex]
        return acc_label

    def getWeight_type_0(self):
        return np.ones(self.cache_size)

    def getWeight_type_1(self, exp=True):
        weight = [.0 for _ in range(self.cache_size)]
        for i in range(self.cache_size):
            label_sum = 0
            for idx in self.cache_trace[i]:
                label_sum += sum(self.true_labels[idx])
            if exp:
                weight[i] = label_sum ** self.a
            else:
                weight[i] = label_sum
        return weight

    def getWeight_type_2(self, exp=True, w_sim=True):
        sim_list = [0. for _ in range(self.cache_size)]
        if self.args.label_type == 2:
            for i in range(self.cache_size):
                if len(self.cache_trace[i]) != 0:
                    activation = self.get_acc_act(self.cache_trace[i])
                    cos_sim = self.get_cos_sim(activation)
                    sim_list[i] = cos_sim
        if self.args.label_type == 0:
            for i in range(self.cache_size):
                if len(self.cache_trace[i]) != 0:
                    acc_label = self.get_acc_label(self.cache_trace[i])
                    cos_sim = self.get_cos_sim(acc_label, vector2=np.ones(self.args.num_classes))
                    sim_list[i] = cos_sim

        # log
        # log_data = np.log(sim_list) * 1000
        # w_sim_lst = softmax(log_data)

        # max
        w_sim_lst = [1. for _ in range(self.cache_size)]
        if w_sim:
            for i in range(self.cache_size):
                temp = 1 / (1 - sim_list[i])
                w_sim_lst[i] = temp

        w_ds_list = self.getWeight_type_1(exp=exp)

        return np.array(w_ds_list) * np.array(w_sim_lst)

    def getWeight(self):
        weight_list = []
        if self.args.W == 0:  # DS
            weight_list = self.getWeight_type_0()
        elif self.args.W == 1:  # DS**a
            weight_list = self.getWeight_type_1()
        elif self.args.W == 2:  # (DS**s)*w_sim
            weight_list = self.getWeight_type_2()
        self.w_var_list.append(np.var(unitization(weight_list)))
        return weight_list

    def c0(self):
        return random.choice(sorted(list(self.idle_clients)))

    def c1(self, model_index, model_version_2):
        idle_client = list(self.idle_clients)
        if model_version_2 == self.args.T - 1:
            return random.choice(idle_client)

        select_range = []
        if np.var(unitization(self.selected_count)) > self.select_var_bound:
            min_value = min(self.selected_count)
            for i in idle_client:
                if self.selected_count[i] == min_value:
                    select_range.append(i)
        else:
            select_range = idle_client

        acc_label = self.get_acc_label(self.model_trace[model_index])
        R = []
        for idx in select_range:
            post_label = acc_label + self.true_labels[idx]
            r1 = self.get_cos_sim(post_label, vector2=np.ones(self.args.num_classes))

            data_size = [sum(self.true_label_count_lists[i]) for i in range(self.cache_size)]
            data_size[model_index] += sum(self.true_labels[idx])
            post_data_size_var = np.var(unitization(data_size))
            r2 = -post_data_size_var
            R.append([idx, r1 + r2])
        R.sort(key=lambda x: x[1], reverse=True)
        best = R[0][0]
        return best

    def c2(self, model_index):
        idle_client = list(self.idle_clients)
        R = []
        acc_act = self.get_acc_act(self.model_trace[model_index])
        for idx in idle_client:
            post_activation = self.activation_labels[idx] + acc_act
            cos_sim = self.get_cos_sim(post_activation)
            R.append([idx, cos_sim])

        w = np.array([1 / i[1] for i in R])
        w /= np.sum(w)
        w = softmax(10 * w)
        return random.choices(R, weights=w)[0][0]

    def c5(self, model_version_2, model_index):
        idle_client = list(self.idle_clients)
        if model_version_2 == self.args.T - 1:
            return random.choice(idle_client)

        temp = [self.selected_count[i] for i in idle_client]
        select_range = []
        if np.var(unitization(self.selected_count)) > self.select_var_bound:
            min_value = min(temp)
            for i in self.idle_clients:
                if self.selected_count[i] == min_value:
                    select_range.append(i)
        else:
            select_range = idle_client

        R = []
        acc_act = self.get_acc_act(self.model_trace[model_index])
        for idx in select_range:
            r1 = 0
            post_activation = self.activation_labels[idx] + acc_act
            r1 = self.get_cos_sim(post_activation)

            r2 = 0
            if self.args.DS == 1:
                data_size = [sum(self.true_label_count_lists[i]) for i in range(self.cache_size)]
                data_size[model_index] += sum(self.true_labels[idx])
                data_size_var = np.var(unitization(data_size))
                r2 = -data_size_var
            R.append([idx, 1 * r1 + r2])
        R.sort(key=lambda x: x[1], reverse=True)
        return R[0][0]

    def select_client(self, model_index, model_version_2, C=None):
        C = self.args.C if C is None else C
        if C == 5:  # 用激活量选
            return self.c5(model_version_2, model_index)
        elif C == 0:  # 随机选
            return self.c0()
        elif C == 1:  # 用hardLabel选
            return self.c1(model_index, model_version_2)
        elif C == 2:  # 用hardLabel选
            return self.c2(model_index)

    def show(self, model_index, VAR_LIST=None, VAR_LIST_2=None):
        print('*' * 80)
        print('Round {:3d}, config:{}'.format(self.global_version, self.name))
        print("time:", self.time)
        print(self.selected_count)
        print("select_var:", np.var(unitization(self.selected_count)))
        print("true_label:{}, true_label_var:{}".format(self.true_label_count_lists[model_index],
                                                        np.var(unitization(self.true_label_count_lists[model_index]))))
        # print("TRUE_LABEL:", self.true_label_count_lists_1[model_index])
        # print("TRUE_LABEL_VAR:", np.var(unitization(self.true_label_count_lists_1[model_index])))
        print(np.mean(self.sim_history))
        self.model_version_diff_history.append(max(self.model_version_list) - min(self.model_version_list))
        print("MV:{}, MV_diff:{}, MV_diff_mean:{}".format(self.model_version_list,
                                                          max(self.model_version_list) - min(self.model_version_list),
                                                          np.mean(self.model_version_diff_history)))
        self.var_list_end.append(np.var(unitization(self.true_label_count_lists[model_index][::])))
        # VAR_LIST_2.append(np.var(unitization(self.true_label_count_lists_1[model_index][::])))
        # self.VAR = np.mean(VAR_LIST)

        print("var_mean_cache:{:.6f}, var_mean_end:{:.6f}, w_var_mean:{:.6f}".
              format(np.mean(self.var_list_cache), np.mean(self.var_list_end), np.mean(self.w_var_list)))
        # print("VAR_MEAN:{:.6f}, VAR_MEAN_1:{:.6f}".format(np.mean(VAR_LIST), np.mean(VAR_LIST_2)))
        ds_var = np.var(unitization(self.getWeight_type_1(exp=False)))
        self.ds_var_history.append(ds_var)
        print("DS_var:{:.6f}, DS_var_mean:{:.6f}".format(ds_var, np.mean(self.ds_var_history)))
        if self.args.label_type == 1:
            print("soft_label", self.soft_label_count_lists[model_index])
            print("soft_label_var:", np.var(unitization(self.soft_label_count_lists[model_index])))

    def train(self):
        self.net_glob.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        init_users = np.random.choice(range(self.args.num_users), m, replace=False)
        for model_index, client_index in enumerate(init_users):
            self.update_list.append([client_index, copy.deepcopy(self.net_glob), model_index,
                                     (0, 0), 0, self.client.getTime(client_index)])
            self.idle_clients.remove(client_index)
            self.selected_count[client_index] += 1
            self.model_trace[model_index].append(client_index)

            self.true_label_count_lists[model_index] += self.true_labels[client_index]

        self.update_list.sort(key=lambda x: x[-1])

        while self.time < self.args.limit_time:
            client_index, model_copy, model_index, \
                (model_version_1, model_version_2), \
                global_model_version, train_time = self.update_list.pop(0)
            for update in self.update_list:
                update[-1] -= train_time
            self.time += train_time

            local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[client_index])
            w = local.train(round=iter, net=model_copy)

            # 判断是否要放入cache
            acc_label = self.get_acc_label(self.model_trace[model_index])
            cos_sim = self.get_cos_sim(acc_label, np.ones(self.args.num_classes))
            self.sim_history.append(cos_sim)
            self.sim_history.sort()
            if model_version_2 > self.args.T // 2 - 1 or self.args.CL == 0 or \
                    self.sim_history.index(cos_sim) > len(self.sim_history) / self.args.CB - 1:
                self.cache[model_index] = copy.deepcopy(w)
                self.cache_trace[model_index] = copy.deepcopy(self.model_trace[model_index])
                self.var_list_cache.append(np.var(unitization(self.true_label_count_lists[model_index][::])))

            if model_version_2 == self.args.T - 1:
                self.global_version += 1
                # cache聚合
                weight_list = self.getWeight()
                new_global_w = Aggregation(self.cache, weight_list)
                self.net_glob.load_state_dict(new_global_w)
                # 把聚合得到的全局模型送入对应的cache
                self.cache[model_index] = copy.deepcopy(new_global_w)
                self.model_version_list[model_index] = self.global_version
                self.show(model_index)
                print(unitization(weight_list))
                self.test()

                self.true_label_count_lists[model_index] = np.zeros(self.args.num_classes)
                self.model_trace[model_index] = []

            next_client = self.select_client(model_index, model_version_2)
            self.selected_count[next_client] += 1
            self.model_trace[model_index].append(next_client)
            self.idle_clients.add(client_index)
            self.idle_clients.remove(next_client)

            if model_version_2 == self.args.T - 1:
                self.update_list.append(
                    [next_client, copy.deepcopy(self.net_glob), model_index, (model_version_1 + 1, 0),
                     self.global_version, self.client.getTime(next_client)])
            else:
                self.update_list.append(
                    [next_client, model_copy, model_index, (model_version_1 + 1, model_version_2 + 1),
                     global_model_version, self.client.getTime(next_client)])

            self.true_label_count_lists[model_index] += self.true_labels[next_client]

            self.update_list.sort(key=lambda x: x[-1])

        self.save_result()

    def train_softLabel(self):
        self.net_glob.train()
        self.update_softLabels()
        m = max(int(self.args.frac * self.args.num_users), 1)
        init_users = np.random.choice(range(self.args.num_users), m, replace=False)
        for model_index, client_index in enumerate(init_users):
            self.update_list.append([client_index, copy.deepcopy(self.net_glob), model_index,
                                     (0, 0), 0, self.client.getTime(client_index)])
            self.idle_clients.remove(client_index)
            self.selected_count[client_index] += 1
            self.model_trace[model_index].append(client_index)
            label_num_count = self.soft_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.soft_labels[client_index][i]

            label_num_count = self.true_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.true_labels[client_index][i]

        self.update_list.sort(key=lambda x: x[-1])

        while self.time < self.args.limit_time:
            client_index, model_copy, model_index, \
                (model_version_1, model_version_2), \
                global_model_version, train_time = self.update_list.pop(0)
            for update in self.update_list:
                update[-1] -= train_time
            self.time += train_time
            self.cache_update_times[model_index] += 1

            local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[client_index])
            w = local.train(round=iter, net=model_copy)

            # 判断是否要放入cache
            var = np.var(unitization(self.soft_label_count_lists[model_index]))
            self.sim_history.append(var)
            self.sim_history.sort()
            if self.args.C != 0:
                if model_version_2 > self.args.T // 2 - 1 or self.args.CL == 0 or \
                        self.sim_history.index(var) > len(self.sim_history) / 4 - 1:
                    self.cache[model_index] = copy.deepcopy(w)
                    self.cache_sample_list[model_index] = self.soft_label_count_lists[model_index][::]
                    self.cache_trace[model_index] = self.model_trace[model_index][::]
            else:
                self.cache[model_index] = copy.deepcopy(w)
                self.cache_sample_list[model_index] = self.soft_label_count_lists[model_index][::]
                self.cache_trace[model_index] = self.model_trace[model_index][::]

            # self.cache[model_index].load_state_dict(w)
            # self.cache_sample_list[model_index] = self.data_label_count_lists[model_index][::]

            if model_version_2 == self.args.T - 1:
                self.show(model_index)
                # cache聚合,然后把聚合得到的全局模型下发给next_client
                self.true_label_count_lists[model_index] = [0 for _ in range(10)]
                self.soft_label_count_lists[model_index] = [0 for _ in range(10)]
                # cache聚合
                # w_locals = [copy.deepcopy(i).state_dict() for i in self.cache]
                weight_list = self.getWeight()
                print(weight_list)
                new_global_w = Aggregation(self.cache, weight_list)
                self.net_glob.load_state_dict(new_global_w)
                # 把聚合得到的全局模型送入对应的cache
                self.cache[model_index] = copy.deepcopy(new_global_w)
                self.global_version += 1
                self.model_version_list[model_index] = self.global_version
                self.model_trace[model_index] = []
                self.test()
                self.update_softLabels()
                self.update_soft_label_count_lists()
                self.update_cache_sample_list()
            elif global_model_version != self.global_version and self.args.GA == 1:
                # 全局模型和cache中的模型聚合,然后把聚合得到的模型下发给next_client
                w_locals = [copy.deepcopy(model_copy).state_dict(), copy.deepcopy(self.net_glob).state_dict()]
                new_local_w = Aggregation(w_locals, [9, 1])
                model_copy.load_state_dict(new_local_w)
                # 把聚合得到的模型送入对应的cache
                if self.args.C != 0:
                    if np.var(unitization(self.soft_label_count_lists[model_index])) < self.label_var_bound \
                            or model_version_2 > self.args.T // 2 - 1 or self.args.CL == 0:
                        self.cache[model_index] = copy.deepcopy(new_local_w)
                else:
                    self.cache[model_index] = copy.deepcopy(new_local_w)

                # self.cache[model_index].load_state_dict(w)
                # self.cache_sample_list[model_index] = self.data_label_count_lists[model_index][::]

            next_client = self.select_client(model_index, model_version_2)
            self.selected_count[next_client] += 1
            self.model_trace[model_index].append(next_client)
            self.idle_clients.remove(next_client)
            self.idle_clients.add(client_index)
            if model_version_2 == self.args.T - 1:
                self.update_list.append(
                    [next_client, copy.deepcopy(self.net_glob), model_index, (model_version_1 + 1, 0),
                     self.global_version, self.client.getTime(next_client)])
            elif global_model_version != self.global_version:
                self.update_list.append(
                    [next_client, model_copy, model_index, (model_version_1 + 1, model_version_2 + 1),
                     self.global_version, self.client.getTime(next_client)])
            else:
                self.update_list.append(
                    [next_client, model_copy, model_index, (model_version_1 + 1, model_version_2 + 1),
                     global_model_version, self.client.getTime(next_client)])

            label_num_count = self.soft_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.soft_labels[next_client][i]

            label_num_count = self.true_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.true_labels[next_client][i]

            self.update_list.sort(key=lambda x: x[-1])

        self.save_result()

    def train_activation(self):
        # VAR_LIST = []
        # VAR_LIST_2 = []
        self.net_glob.train()
        self.update_activation()
        m = max(int(self.args.frac * self.args.num_users), 1)
        self.comm += m
        init_users = np.random.choice(range(self.args.num_users), m, replace=False)
        for model_index, client_index in enumerate(init_users):
            self.update_list.append([client_index, copy.deepcopy(self.net_glob), model_index,
                                     (0, 0), 0, self.client.getTime(client_index)])
            self.idle_clients.remove(client_index)
            self.selected_count[client_index] += 1
            self.model_trace[model_index].append(client_index)

            self.true_label_count_lists[model_index] += self.true_labels[client_index]

            # label_num_count = self.true_label_count_lists_1[model_index]
            # for i in range(len(label_num_count)):
            #     label_num_count[i] += self.true_labels_1[client_index][i]

        self.update_list.sort(key=lambda x: x[-1])

        while self.time < self.args.limit_time:
            client_index, model_copy, model_index, \
                (model_version_1, model_version_2), \
                global_model_version, train_time = self.update_list.pop(0)
            for update in self.update_list:
                update[-1] -= train_time
            self.time += train_time
            self.comm += 1
            self.cache_update_times[model_index] += 1

            local = LocalUpdate_FedAvg(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[client_index])
            w = local.train(round=self.global_version, net=model_copy)

            # 判断是否要放入cache
            post_activation = self.get_acc_act(self.model_trace[model_index])
            post_cos_sim = self.get_cos_sim(post_activation)
            self.sim_history.append(post_cos_sim)
            self.sim_history.sort()
            if model_version_2 > self.args.T // 2 - 1 or self.args.CL == 0 or \
                    (self.sim_history.index(post_cos_sim) + 1) / len(self.sim_history) > (1-self.args.CB):
                self.cache[model_index] = copy.deepcopy(w)
                self.cache_trace[model_index] = copy.deepcopy(self.model_trace[model_index])
                self.var_list_cache.append(np.var(unitization(self.true_label_count_lists[model_index][::])))
                # VAR_LIST.append(np.var(unitization(self.true_label_count_lists_1[model_index][::])))

            if model_version_2 == self.args.T - 1:
                self.global_version += 1
                # cache聚合
                weight_list = self.getWeight()
                new_global_w = Aggregation(self.cache, weight_list)
                self.net_glob.load_state_dict(new_global_w)
                # 把聚合得到的全局模型送入对应的cache
                self.cache[model_index] = copy.deepcopy(new_global_w)
                self.model_version_list[model_index] = self.global_version
                self.show(model_index)
                print(unitization(weight_list))
                self.test()

                self.true_label_count_lists[model_index] = np.zeros(self.args.num_classes)
                self.model_trace[model_index] = []

                if self.global_version % self.args.CF == 0:
                    self.comm += self.args.num_users
                    self.update_activation()

            next_client = self.select_client(model_index, model_version_2)
            self.selected_count[next_client] += 1
            self.model_trace[model_index].append(next_client)
            self.idle_clients.add(client_index)
            self.idle_clients.remove(next_client)

            self.comm += 1
            if model_version_2 == self.args.T - 1:
                self.update_list.append(
                    [next_client, copy.deepcopy(self.net_glob), model_index, (model_version_1 + 1, 0),
                     self.global_version, self.client.getTime(next_client)])
            else:
                self.update_list.append(
                    [next_client, model_copy, model_index, (model_version_1 + 1, model_version_2 + 1),
                     global_model_version, self.client.getTime(next_client)])

            self.true_label_count_lists[model_index] += self.true_labels[next_client]

            self.update_list.sort(key=lambda x: x[-1])

        self.save_result()

    def simulate(self):
        var_list = []
        var_list_2 = []
        w_var_list = []
        version_var_list = []
        m = max(int(self.args.frac * self.args.num_users), 1)
        init_users = np.random.choice(range(self.args.num_users), m, replace=False)
        for model_index, client_index in enumerate(init_users):
            self.update_list.append([client_index, model_index,
                                     (0, 0), 0, self.client.getTime(client_index)])
            self.idle_clients.remove(client_index)
            self.selected_count[client_index] += 1
            self.model_trace[model_index].append(client_index)
            label_num_count = self.true_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.true_labels[client_index][i]
        self.update_list.sort(key=lambda x: x[-1])

        while self.time < self.args.limit_time:
            client_index, model_index, \
                (model_version_1, model_version_2), \
                global_model_version, train_time = self.update_list.pop(0)
            for update in self.update_list:
                update[-1] -= train_time
            self.time += train_time
            self.cache_update_times[model_index] += 1
            # self.model_version_list[model_index] += 0.1

            # 判断是否要放入cache
            if np.var(unitization(
                    self.true_label_count_lists[model_index])) < self.label_var_bound \
                    or model_version_2 > self.args.T // 2 - 1 or self.args.CL == 0:
                self.cache_sample_list[model_index] = self.true_label_count_lists[model_index][::]
                self.cache_trace[model_index] = self.model_trace[model_index][::]
                var_list.append(np.var(unitization(self.cache_sample_list[model_index])))

            if model_version_2 == self.args.T - 1:
                # self.show(model_index)
                # cache聚合,然后把聚合得到的全局模型下发给next_client
                var_list_2.append(np.var(unitization(self.true_label_count_lists[model_index])))
                self.true_label_count_lists[model_index] = [0 for _ in range(self.args.num_classes)]
                # cache聚合
                weight_list = self.getWeight_type_0()
                w_var_list.append(np.var(unitization(weight_list)))
                # print(weight_list)
                # 把聚合得到的全局模型送入对应的cache
                version_var_list.append(max(self.model_version_list) - min(self.model_version_list))
                self.global_version += 1
                self.model_version_list[model_index] = self.global_version
                self.model_trace[model_index] = []

            next_client = self.select_client(model_index, model_version_2)
            self.selected_count[next_client] += 1
            self.model_trace[model_index].append(next_client)
            self.idle_clients.add(client_index)
            self.idle_clients.remove(next_client)
            if model_version_2 == self.args.T - 1:
                self.update_list.append(
                    [next_client, model_index, (self.global_version, 0),
                     self.global_version, self.client.getTime(next_client)])
            else:
                self.update_list.append(
                    [next_client, model_index, (model_version_1, model_version_2 + 1),
                     global_model_version, self.client.getTime(next_client)])

            label_num_count = self.true_label_count_lists[model_index]
            for i in range(len(label_num_count)):
                label_num_count[i] += self.true_labels[next_client][i]
            self.update_list.sort(key=lambda x: x[-1])
        print(self.selected_count)
        print(sum(var_list) / len(var_list))
        var_list.sort()
        print(var_list[len(var_list) // 2])
        print(max(var_list), var_list[len(var_list) * 3 // 4], var_list[len(var_list) // 4], min(var_list))
        var_list = var_list_2
        print(sum(var_list) / len(var_list))
        var_list.sort()
        print(var_list[len(var_list) // 2])
        print(max(var_list), var_list[len(var_list) * 3 // 4], var_list[len(var_list) // 4], min(var_list))

        print("w_var_mean: ", np.mean(w_var_list))
        print("select_var: ", np.var(unitization(self.selected_count)))

        print(self.select_var_count, self.label_var_count)
        print(self.label_var_bound)
        # var_list = client_var_list
        # print(sum(var_list) / len(var_list))
        # var_list.sort()
        # print(var_list[len(var_list) // 2])
        # print(max(var_list), var_list[len(var_list) * 3 // 4], var_list[len(var_list) // 4], min(var_list))

        var_list = version_var_list
        print(sum(var_list) / len(var_list))
        var_list.sort()
        print(var_list[len(var_list) // 2])
        print(max(var_list), var_list[len(var_list) * 3 // 4], var_list[len(var_list) // 4], min(var_list))


def unitization(x):
    n = np.sum(x)
    if n == 0:
        return x
    else:
        return x / n


def power(num, index):
    if num > 0:
        return num ** index
    return -((-num) ** index)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
