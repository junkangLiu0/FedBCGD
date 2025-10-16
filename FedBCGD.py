import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import time
import random
from math import exp
from copy import deepcopy
import ray
import argparse
from torchsummary import summary
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from torch.autograd import Variable
from models.resnet import ResNet18,ResNet50,ResNet10
from models.resnet_bn import ResNet18BN,ResNet50BN,ResNet10BN,ResNet34BN
# 加入存档，log
#python  FedBCGD.py --alg FedBCGD --lr 0.1 --data_name CIFAR100 --alpha_value 0.6  --epoch 1001  --extname CIFAR100 --lr_decay 0.998 --CNN lenet5 --E 5 --batch_size 50  --gpu 8 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 1

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=1000, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=50, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedAvg', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='1', type=float, help='lr_decay')
parser.add_argument('--data_name', default='MNIST', type=str, help='lr_decay')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--gamma', default=0.45, type=float, help=' for mom_step')
parser.add_argument('--num_gpus_per', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='lenet5', type=str, help=' for mom_step')

parser.add_argument('--p', default=2, type=float, help=' for mom_step')
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
parser.add_argument('--print', default=0, type=int, help=' for mom_step')
parser.add_argument("--rho", type=float, default=0.05, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--adaptive", type=bool, default=True, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--optimizer', default='SGD', type=str, help='adam')
parser.add_argument("--preprint", type=int, default=10, help="")
parser.add_argument('--block', default=5, type=float, help='block')
parser.add_argument('--B', default=6, type=float, help=' for mom_step')





args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
# num_gpus_per = 1
data_name = args.data_name
CNN = args.CNN

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if data_name == 'CIFAR10':

    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=False,
        transform=transform_train)
elif data_name == 'EMNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        split='byclass',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )

elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train)
elif data_name == 'MNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        #split='mnist',
        split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )


def get_data_loader(pid, data_idx, batch_size, data_name):
    """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
    sample_chosed = data_idx[pid]
    train_sampler = SubsetRandomSampler(sample_chosed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler)
    return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='byclass', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'MNIST':
        #test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        test_dataset = datasets.EMNIST("./data", split='balanced', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader
   
def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform_train)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'EMNIST':
        train_dataset = datasets.EMNIST("./data", split='byclass', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_train)
    elif data_name == 'MNIST':
        #test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        train_dataset = datasets.EMNIST("./data", split='balanced', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return train_loader


def evaluate(model, test_loader,train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss=0
    train_loss=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(outputs, target)
            
        for batch_idx, (data, target) in enumerate(train_loader):
            data_train = data.to(device)
            target_train = target.to(device)
            outputs_train = model(data_train)
            train_loss+= criterion(outputs_train, target_train)
    return 100. * correct / total,test_loss/ len(test_loader),train_loss/ len(train_loader)
    
def evaluate2(model, test_loader,train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target =target.to(device)               
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total,torch.tensor(0),torch.tensor(0)





class ConvNet_EMNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet_MNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 47)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)



class Lenet5(nn.Module):

    def __init__(self, num_classes=100):
        super(Lenet5, self).__init__()
        self.n_cls = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

if CNN == 'lenet5':
    def ConvNet():
        return Lenet5(num_classes=10)
    def ConvNet100():
        return Lenet5(num_classes=100)
if CNN == 'resnet18':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10, l2_norm=False):
            return ResNet18BN(num_classes=10)
        def ConvNet100(num_classes=100, l2_norm=False):
            return ResNet18BN(num_classes=100)
        def ConvNet200(num_classes=200, l2_norm=False):
            return ResNet18BN(num_classes=200)
    if args.normalization=='GN':
        def ConvNet(num_classes=10):
            return ResNet18(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet18(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet18(num_classes=200)

if CNN == 'resnet10':
    if args.normalization=='BN':
        def ConvNet(num_classes=10):
            return ResNet10BN(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10BN(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10BN(num_classes=200)
    if args.normalization=='GN':
        def ConvNet(num_classes=10):
            return ResNet10(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10(num_classes=200)




@ray.remote
class ParameterServer(object):
    def __init__(self, lr, alg, tau, selection, data_name,num_workers):
        if data_name == 'CIFAR10':
            self.model = ConvNet()
        elif data_name == 'EMNIST':
            self.model = ConvNet_EMNIST()
        elif data_name == 'CIFAR100':
            self.model = ConvNet100()
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.momen_v = None
        #self.gamma = 0.9
        self.gamma=args.gamma
        #self.gamma = 0.5
        self.beta = 0.99  # 论文是0.99
        self.alg = alg
        self.num_workers = num_workers
  
        self.lr_ps = lr

        self.ps_c = None
        self.c_all = None
        # 上一代的c
        self.c_all_pre = None
        self.tau = tau
        self.selection = selection
        self.cnt = 0
        self.alpha = None
        self.h = {}
        self.momen_m={}
        if args.CNN == 'lenet5':
            self.block_share = ['fc3.weight', 'fc3.bias', 'fc2.weight', 'fc2.bias']
        if args.CNN == 'resnet18':
            self.block_share = ['linear.bias', 'linear.weight']



    def set_pre_c(self, c):
        self.c_all_pre = c

    def apply_weights_avg(self, num_workers, *weights):
        '''
        weights: delta_w
        '''
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        for weight in weights:
            for k, v in weight.items():
                if k in self.block_share:

                    if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                        sum_weights[k] += v / (num_workers * self.selection)
                    # sum_weights[k]+=v / num_workers
                    else:
                        sum_weights[k] = v / (num_workers * self.selection)
                else:
                    if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                        sum_weights[k] += v / (num_workers * self.selection / args.block)
                        # sum_weights[k]+=v / num_workers
                    else:
                        sum_weights[k] = v / (num_workers * self.selection / args.block)
        for k, v in sum_weights.items():  # w = w + delta_w
            sum_weights[k] = ps_w[k] + sum_weights[k]
        self.model.set_weights(sum_weights)
        return self.model.get_weights()

    def apply_weights_moment(self, num_workers, *weights):
        self.gamma=0.45
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                # if  'linear' in k or k=='conv1.weight' or k=='bn1.weight' or k=='bn1.bias' :
                if k in self.block_share:
                    if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                        sum_weights[k] += v / (num_workers * selection)
                    # sum_weights[k]+=v / num_workers
                    else:
                        sum_weights[k] = v / (num_workers * selection)
                else:
                    if k in sum_weights.keys():
                        sum_weights[k] += v / (num_workers * selection / args.block)
                    else:
                        sum_weights[k] = v / (num_workers * selection / args.block)
        weight_ps = self.model.get_weights()
        # for k,v in weight_ps.items():
        #     sum_weights[k]-=v
        if not self.momen_v:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                # if k=='fc3.weight' or k=='fc3.bias' or k=='fc2.weight' or k=='fc2.bias':
                #  self.momen_v[k] = 0.9 * v + sum_weights[k]
                # else:
                self.momen_v[k] = self.gamma * v + sum_weights[k]
                # self.momen_v[k] =  v + sum_weights[k]*0.15
                # self.momen_v[k] = v + sum_weights[k]
        seted_weight = {}
        for k, v in weight_ps.items():
            seted_weight[k] = v + self.momen_v[k]
        self.model.set_weights(seted_weight)
        return self.model.get_weights()

    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.apply_weights_avg,
            'FedMoment': self.apply_weights_moment,
            'SCAFFOLD': self.apply_weights_avg,
            'SCAFFOLDM': self.apply_weights_moment,
            'IGFL': self.apply_weights_avg,
            #'FedAdam': self.apply_weights_adam,
            'FedDyn': self.apply_weights_avg,
            'FedCM':self.apply_weights_avg,
            'FedDC': self.apply_weights_moment,
            'FedBCGD': self.apply_weights_avg,
            'FedBCGD+': self.apply_weights_moment,
        }

    def apply_weights_func(self, alg, num_workers, *weights):
        self.load_dict()
        return self.func_dict.get(alg, None)(num_workers, *weights)
        


    def apply_ci(self, alg, num_workers, *cis):
        '''
        平均所有的ds发来的sned_ci: delta_ci
        '''
        if 'atte' in alg:
            # 先将当前状态传给self.c_all_pre, 再平均所有的ds发来的ci，更新self.ps_c
            self.set_pre_c(self.c_all)
            self.c_all = cis

        sum_c = {}  # delta_c :sum_c
        for ci in cis:
            for k, v in ci.items():
                if k=='fc3.weight' or k=='fc3.bias' or k=='fc2.weight' or k=='fc2.bias':
                
                    if k in sum_c.keys():
                        sum_c[k] += v/(selection *self.num_workers)
                    else:
                        sum_c[k] = v/(selection *self.num_workers)
                else:
                    if k in sum_c.keys():
                        sum_c[k] += v/(num_workers*selection/5)
                    else:
                        sum_c[k] = v/(num_workers*selection/5)
        if self.ps_c == None:
            self.ps_c = sum_c
            return self.ps_c
        for k, v in self.ps_c.items():
            if alg in {'FedCM','FedDyn','IGFL_prox','FedAGM','FedDC','IGFL'}:
                self.ps_c[k] = v +  sum_c[k]
            else:
                self.ps_c[k] = v + 0.1 * sum_c[k]
        return self.ps_c

    def get_weights(self):
        return self.model.get_weights()

    def get_ps_c(self):
        return self.ps_c

    def get_state(self):
        return self.ps_c, self.c_all

    def set_state(self, c_tuple):
        self.ps_c = c_tuple[0]
        self.c_all = c_tuple[1]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_attention(self):
        return self.alpha


@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'EMNIST':
            # self.model = SCAFNET().to(device)
            self.model = ConvNet_EMNIST().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST().to(device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx

        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        if args.CNN == 'lenet5':
            self.block_share = ['fc3.weight', 'fc3.bias', 'fc2.weight', 'fc2.bias']
        if args.CNN == 'resnet18':
            self.block_share = ['linear.bias', 'linear.weight']

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]
        
    def state_hi_loader(self, index):
        if not hi_dict.get(index):
            return
        self.hi = hi_dict[index]

    def state_Li_loader(self, index):
        if not Li_dict.get(index):
            return
        self.Li = Li_dict[index]

    def get_train_loss(self):
        return self.loss

    def update_fedavg(self, weights, E, index, lr, index2):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        optim_param = []
        i = 0
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        self.loss = loss.item()

        delta_w = deepcopy(self.model.get_weights())
        # '''
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        for k, v in self.model.get_weights().items():
            if k in index2:

                delta_w[k] = v - weights[k]
            else:
                delta_w[k] = zero_weight[k]
                # '''
        return delta_w

    def update_scafplus(self, weights, E, index, ps_c, lr, index2):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        optim_param = []
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] + ps_c[k])  # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        send_ci = deepcopy(self.model.get_weights())
        ci = deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator)) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        # '''
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        for k, v in self.model.get_weights().items():
            if k in index2:

                delta_w[k] = v - weights[k]
            else:
                delta_w[k] = zero_weight[k]
        for k, v in self.model.get_weights().items():
            if k in index2:

                send_ci[k] = - ps_c[k] + self.ci[k]
            else:
                send_ci[k] = zero_weight[k]
                # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci



        
        
        
        




    def update_SCAFFOLDM(self, weights, E, index, ps_c,lr,index2):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        optim_param = []
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] + ps_c[k])  # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        send_ci = deepcopy(self.model.get_weights())
        ci = deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator)) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        # '''
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        for k, v in self.model.get_weights().items():
            if k in index2:

                delta_w[k] = v - weights[k]
            else:
                delta_w[k] = zero_weight[k]
        for k, v in self.model.get_weights().items():
            if k in index2:

                send_ci[k] = - ps_c[k] + self.ci[k]
            else:
                send_ci[k] = zero_weight[k]
                # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci
        

   



    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,  # base FedAvg
            'FedMoment': self.update_fedavg,  # add moment
            #'SCAFFOLD': self.update_scaf,  # scaf
            'FedAdam': self.update_fedavg,  # FedAdam
            #'FedDyn':self.update_fedDyn,
            #'FedCM':self.update_FedCM,
            #'FedDC':self.update_FedDC,
            'SCAFFOLDM':self.update_SCAFFOLDM,
            'FedBCGD': self.update_fedavg,
            'FedBCGD+':self.update_SCAFFOLDM,
        

        }

    def update_func(self, alg, weights, E, index,insex2,lr, ps_c=None):
        self.load_dict()
        if alg in {'SCAFFOLD', 'IGFL','FedDyn','FedCM','FedDC','SCAFFOLDM','FedBCGD+'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr,insex2)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr,insex2)

def set_random_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    set_random_seed(seed=42)

    # 获取args
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    gamma=args.gamma
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay=args.lr_decay
    
    hi_dict = {}
    Li_dict = {}

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value)
    c_dict = {}  # state dict
    assert alg in {
        'FedBCGD',
        'FedBCGD+',
        'FedAvg',
        'FedMoment',
        'SCAFFOLD',
        'IGFL',
        'FedAdam',
        'FedDyn',
        'FedCM',
        'FedDC',
        'FedAGM',
        'SCAFFOLDM'

    }
    #  配置logger
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                 .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'EMINIST':
        nums_cls = 62
    if data_name == 'MNIST':
        nums_cls = 47

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = 500
    if data_name == 'EMINIST':
        nums_sample = 6979
    if data_name == 'MNIST':
        nums_sample = 500
    if data_name == 'CIFAR100':
        nums_sample = 500

    data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
    logger.info('std:{}'.format(std))
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)
    ps = ParameterServer.remote(lr_ps, alg, tau, selection, data_name,num_workers)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'EMNIST':
        # model = SCAFNET().to(device)
        model = ConvNet_EMNIST().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    elif data_name == 'MNIST':
        # model = SCAFNET().to(device)
        model = ConvNet_MNIST().to(device)
    if check:
        model_CKPT = torch.load(checkpoint_path)
        print('loading checkpoint!')
        model.load_state_dict(model_CKPT['state_dict'])
        c_dict = model_CKPT['c_dict']
        Li_dict = model_CKPT['Li_dict']
        hi_dict = model_CKPT['hi_dict']
        ps_state = model_CKPT['ps_state']
        ray.get(ps.set_state.remote(ps_state))
        epoch_s = model_CKPT['epoch']
        data_idx = model_CKPT['data_idx']
    else:
        epoch_s = 0
        # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection))]
    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr,alpha_value,alpha,gamma))
    # logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")


    ps.set_weights.remote(model.get_weights())
    current_weights = ps.get_weights.remote()
    ps_c = ps.get_ps_c.remote()

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    zero=model.get_weights()
    for k, v in model.get_weights().items():
        zero[k]=zero[k]-zero[k]
    ps_c=deepcopy(zero)



    for epochidx in range(epoch_s, epoch):
        index = np.arange(num_workers)  # 100
        lr=lr*lr_decay
        np.random.shuffle(index)

        if args.CNN == 'lenet5':
            block1 = ['conv1.weight', 'conv1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            block2 = ['conv2.weight', 'conv2.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            block3 = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            block4 = ['fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            block5 = ['fc3.weight', 'fc3.bias', 'fc2.weight', 'fc2.bias']
            block_share = ['fc3.weight', 'fc3.bias', 'fc2.weight', 'fc2.bias']
            index2 = [block1, block2, block3, block4, block5]
            index2 = index2 + index2
            # print(block_share)
        if args.CNN == 'resnet18':
            block1 = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight',
                      'layer1.0.bn1.bias', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight',
                      'layer1.0.bn2.bias', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                      'layer1.1.conv2.weight'
                , 'layer1.1.bn2.weight', 'layer1.1.bn2.bias']
            block2 = ['layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.conv2.weight',
                      'layer2.0.bn2.weight'
                , 'layer2.0.bn2.bias', 'layer2.0.shortcut.0.weight', 'layer2.0.shortcut.1.weight',
                      'layer2.0.shortcut.1.bias', 'layer2.1.conv1.weight'
                , 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight',
                      'layer2.1.bn2.bias']
            block3 = ['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.conv2.weight',
                      'layer3.0.bn2.weight'
                , 'layer3.0.bn2.bias', 'layer3.0.shortcut.0.weight', 'layer3.0.shortcut.1.weight',
                      'layer3.0.shortcut.1.bias', 'layer3.1.conv1.weight'
                , 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight',
                      'layer3.1.bn2.bias']
            block4 = ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.conv2.weight',
                      'layer4.0.bn2.weight'
                , 'layer4.0.bn2.bias', 'layer4.0.shortcut.0.weight', 'layer4.0.shortcut.1.weight',
                      'layer4.0.shortcut.1.bias'
                      ]
            block5 = ['layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.conv2.weight',
                      'layer4.1.bn2.weight', 'layer4.1.bn2.bias']
            block_share = ['linear.bias', 'linear.weight']
            block1 = block1 + block_share
            block2 = block2 + block_share
            block3 = block3 + block_share
            block4 = block4 + block_share
            block5 = block5 + block_share
            index2 = [block1, block2, block3, block4, block5]
            index2 = index2 + index2
            np.random.shuffle(index2)

        if alg in {'SCAFFOLD', 'FedCM','FedDyn','FedAGM','IGFL','SCAFFOLDM','FedBCGD+',}:
            weights_and_ci = [
                worker.update_func.remote(alg, current_weights, E, idx,idx2,lr,ps_c) for worker, idx,idx2 in zip(workers, index,index2)
            ]
            weights_and_ci = ray.get(weights_and_ci)
            weights = [w for w, ci in weights_and_ci]
            ci = [ci for w, ci in weights_and_ci]

            ps_c = ps.apply_ci.remote(alg, num_workers, *ci)
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
        elif alg in {'FedAvg','FedMoment','FedAdam','FedBCGD',}:
            weights = [
                worker.update_func.remote(alg, current_weights, E, idx,idx2,lr) for worker, idx,idx2 in zip(workers, index,index2)
            ]
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)


        if epochidx % 10 == 0:
            # Evaluate the current model.
            test_loss=0
            train_loss=0
            model.set_weights(ray.get(current_weights))
            accuracy,test_loss,train_loss = evaluate(model, test_loader,train_loader)            
            test_loss=test_loss.to('cpu')
            loss_train_median=train_loss.to('cpu')        
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                ps_state = ps.get_state.remote()
                torch.save({'epoch': epochidx + 1, 'state_dict': model.state_dict(), 'c_dict': c_dict,
                            'ps_state': ray.get(ps_state), 'data_idx': data_idx},
                           checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info("Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}".format(epochidx, accuracy,
                                                                                                     loss_train_median,test_loss,
                                                                                                     no_improve))
                                                                                                     
            print("Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}".format(epochidx, accuracy,
                                                                                               loss_train_median,test_loss,
                                                                                               no_improve))
            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)

    # the last is time cost
    result_list_loss.append(endtime - start)

    result_loss = np.array(result_list_loss)
    test_list_loss=np.array(test_list_loss)

    save_name = './plot/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}'.format(alg, E, num_workers, epoch,
                                                                                          lr, alpha_value, selection,alpha,
                                                                                          extra_name,gamma)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss,test_list_loss))
    ray.shutdown()