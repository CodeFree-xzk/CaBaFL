import copy

import numpy as np
import torch
import wandb
from loguru import logger

from models import LocalUpdate_WKAFL, test_img
from utils.Clients import Clients


class WKAFL:
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users):
        self.update_list = []
        self.args = args
        self.net_glob = net_glob
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.clients = Clients(args, dict_users)
        self.train_time_list = self.clients.train_time
        self.time = 0
        self.idle_set = set(list(range(args.num_users)))
        self.acc = []
        self.max_avg = 0
        self.max_std = 0
        self.m = int(args.num_users * args.frac)
        self.args.K = min(10, max(3, int(self.m * 0.5)))

        group = "IID"
        if args.iid == 0:
            group = str(args.data_beta)
        wandb.init(project="FedTrace", name=group + "_" + args.algorithm, group=group,
                   tags=[str(args.model), args.dataset])
        wandb.log({'acc': 0, 'max_avg': 0, 'time': 0})
        ##################################获取模型层数和各层的形状#############
        Layers_num = 0
        Layers_shape = []
        Layers_nodes = []
        for idx, params in enumerate(net_glob.parameters()):
            Layers_num = idx
            Layers_shape.append(params.shape)
            Layers_nodes.append(params.numel())
        self.Layers_num, self.Layers_shape, self.Layers_nodes = Layers_num + 1, Layers_shape, Layers_nodes

    def test(self, model):
        acc_test, loss_test = test_img(model, self.dataset_test, self.args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        self.acc.append(acc_test.item())
        if len(self.acc) >= 10:
            avg = sum(self.acc[len(self.acc) - 10::]) / 10
            if avg > self.max_avg:
                self.max_avg = avg
                self.max_std = np.std(self.acc[len(self.acc) - 10::])
        print("max_avg:{:.2f}, max_std:{:.2f}".format(self.max_avg, self.max_std))
        wandb.log({'acc': acc_test.item(), 'max_avg': self.max_avg, 'time': self.time})

    ##################################设置各层的梯度为0#####################
    def ZerosGradients(self, Layers_shape):
        ZeroGradient = []
        for i in range(len(Layers_shape)):
            ZeroGradient.append(torch.zeros(Layers_shape[i]))
        return ZeroGradient

    ################################调整学习率###############################
    def lr_adjust(self, tau):
        tau = 0.01 * tau + 1
        lr = self.args.lr / tau
        return lr

    #################################计算范数################################
    def L_norm(self, Tensor):
        norm_Tensor = torch.tensor([0.])
        norm_Tensor = norm_Tensor.cuda(self.args.gpu)
        for i in range(len(Tensor)):
            norm_Tensor += Tensor[i].float().norm() ** 2
        return norm_Tensor.sqrt()

    ################################# 计算角相似度 ############################
    def similarity(self, user_Gradients, yun_Gradients):
        sim = torch.tensor([0.])
        sim = sim.cuda(self.args.gpu)
        for i in range(len(user_Gradients)):
            user_Gradients[i] = user_Gradients[i].cuda(self.args.gpu)
            yun_Gradients[i] = yun_Gradients[i].cuda(self.args.gpu)
            sim = sim + torch.sum(user_Gradients[i] * yun_Gradients[i])
        if self.L_norm(user_Gradients) == 0:
            print('梯度为0.')
            sim = torch.tensor([1.])
            return sim
        sim = sim / (self.L_norm(user_Gradients) * self.L_norm(yun_Gradients))
        return sim

    ################################ 定义剪裁 #################################
    def TensorClip(self, Tensor, ClipBound):
        norm_Tensor = self.L_norm(Tensor)
        if ClipBound < norm_Tensor:
            for i in range(self.Layers_num):
                Tensor[i] = Tensor[i] * ClipBound / norm_Tensor
        return Tensor

    #################################聚合####################################
    def aggregation(self, Collect_Gradients, K_Gradients, weight, Layers_shape, args, Clip=False):
        sim = torch.zeros([args.K])
        Gradients_Total = torch.zeros([args.K + 1])
        for i in range(args.K):
            Gradients_Total[i] = self.L_norm(K_Gradients[i])
        Gradients_Total[args.K] = self.L_norm(Collect_Gradients)
        for i in range(args.K):
            sim[i] = self.similarity(K_Gradients[i], Collect_Gradients)
        index = (sim > args.threshold)
        if sum(index) == 0:
            print("相似度均较低")
            return Collect_Gradients
        Collect_Gradients = self.ZerosGradients(Layers_shape)

        totalSim = []
        Sel_Gradients = []
        for i in range(args.K):
            if sim[i] > args.threshold:
                totalSim.append((torch.exp(sim[i] * 50) * weight[i]).tolist())
                Sel_Gradients.append(K_Gradients[i])
        totalSim = torch.tensor(totalSim)
        totalSim = totalSim / torch.sum(totalSim)
        for i in range(len(totalSim)):
            Gradients_Sample = Sel_Gradients[i]
            if Clip:
                standNorm = Gradients_Total[len(Gradients_Total) - 1]
                Gradients_Sample = self.TensorClip(Gradients_Sample, args.CB2 * standNorm)
            for j in range(len(K_Gradients[i])):
                Collect_Gradients[j] = Collect_Gradients[j].cuda(self.args.gpu)
                Collect_Gradients[j] = Collect_Gradients[j] + Gradients_Sample[j] * totalSim[i]
        return Collect_Gradients

    @logger.catch
    def train(self):
        model = self.net_glob
        e = torch.exp(torch.tensor(1.))
        itr = 1

        for client_index in range(self.m):
            self.idle_set.remove(client_index)
            self.update_list.append(
                [client_index, copy.deepcopy(model), itr, self.clients.getTime(client_index)])
        Collect_Gradients = None
        while self.time < self.args.limit_time:
            # 生成与模型梯度结构相同的元素=0的列表
            K_tau = []
            K_Gradients = []

            self.update_list.sort(key=lambda x: x[-1])
            got_updates = self.update_list[0:self.args.K]
            self.update_list = self.update_list[self.args.K::]
            self.time += got_updates[-1][-1]
            for update in self.update_list:
                update[-1] -= got_updates[-1][-1]

            for client_index, model, model_version, train_time in got_updates:
                local = LocalUpdate_WKAFL(args=self.args, dataset=self.dataset_train,
                                          idxs=self.dict_users[client_index])
                self.idle_set.add(client_index)
                Gradients_Sample = local.train(round=iter, net=model)
                K_tau.append(itr - model_version + 1)
                if itr > 1:
                    for j in range(self.Layers_num):
                        Gradients_Sample[j] = Gradients_Sample[j] + self.args.alpha * Collect_Gradients[j]
                K_Gradients.append(self.TensorClip(Gradients_Sample, self.args.CB1))

            Collect_Gradients = self.ZerosGradients(self.Layers_shape)
            K_tau = torch.tensor(K_tau) * 1.
            _, index = torch.sort(K_tau)
            weight = (e / 2) ** (-K_tau)
            if torch.sum(weight) == 0:
                print("延时过大。")
                for i in range(self.Layers_num):
                    weight[index[0]] = 1.
                    Collect_Gradients = K_Gradients[index[0]]
            else:
                weight = weight / torch.sum(weight)
                for i in range(self.args.K):
                    Gradients_Sample = K_Gradients[i]
                    for j in range(self.Layers_num):
                        Gradients_Sample[j] = Gradients_Sample[j].cuda(self.args.gpu)
                        Collect_Gradients[j] = Gradients_Sample[j] * weight[i]

            if itr < self.args.stageTwo:
                Collect_Gradients = self.aggregation(Collect_Gradients, K_Gradients, weight, self.Layers_shape,
                                                     self.args)
            elif itr > 100:
                Collect_Gradients = self.aggregation(Collect_Gradients, K_Gradients, weight, self.Layers_shape,
                                                     self.args, Clip=True)

            lr = self.lr_adjust(torch.min(K_tau))
            for grad_idx, params_sever in enumerate(model.parameters()):
                params_sever.data.add_(-lr, Collect_Gradients[grad_idx])

            idle_list = list(self.idle_set)
            idxs_users = np.random.choice(idle_list, self.args.K, replace=False)
            for client_index in idxs_users:
                self.update_list.append([client_index, copy.deepcopy(model), itr, self.clients.getTime(client_index)])
                self.idle_set.remove(client_index)

            print('*' * 80)
            print('Round {:3d}'.format(itr))
            print("time:", self.time)
            self.test(model)
            itr += 1
