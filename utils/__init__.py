import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from utils.als import AdaptiveLabelSmoothingLoss
import scipy.stats
import random

from . import few_shot
from . import als
from . import sam
import pdb

# torch.cuda.set_device(0)

_log_path = None

# cifar_mean = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
# cifar_std = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]

# cifar_mu = torch.tensor(cifar_mean).view(3,1,1).cuda()
# cifar_std = torch.tensor(cifar_std).view(3,1,1).cuda()

def inverse_normalize_cifar(x) :
    x[:,0,:,:] = x[:,0,:,:] * cifar_std[0] + cifar_mu[0]
    x[:,1,:,:] = x[:,1,:,:] * cifar_std[1] + cifar_mu[1]
    x[:,2,:,:] = x[:,2,:,:] * cifar_std[2] + cifar_mu[2]
    return x

def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))# remove the last / in a string
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def compute_logits(feat, proto, metric='dot', temp=1.0):
    #feat: x_query, 4,75,512; proto: x_shot, 4,5,512
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
            #batch matrix-matrix product; proto.permute(0, 2, 1): 4,512,5
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def sample_classes(total, num, exclude):
    lst = list(range(0,exclude))+list(range(exclude+1,total))
    sampled = random.sample(lst, num-1)
    sampled.append(int(exclude))
    return sampled

def attack_pgd(model, dataset, X, y, epsilon, alpha, attack_iters, device, restarts=1, norm="l_inf", early_stop=False, fgsm_init=None, als=False, history_label = None, als_loss = None, flooding = None):
    max_loss = torch.zeros(y.shape[0]).cuda(device)
    max_delta = torch.zeros_like(X).cuda(device)
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda(device)
        if attack_iters>1 or fgsm_init=='random':
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for iteration in range(attack_iters):
            output = model(dataset.normalize_raw(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                #early stop==True and all preds are wrong
                break
            if als:
                loss = als_loss(output,y,history_label=history_label)
                if flooding:
                    loss = (loss - flooding).abs() + flooding
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            if als:
                all_loss = als_loss(model(dataset.normalize_raw(X+delta)), 
                        y, history_label=history_label)
                if flooding:
                    loss = (loss - flooding).abs() + flooding
            else:
                all_loss = F.cross_entropy(model(dataset.normalize_raw(X+delta)), 
                        y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
            max_delta = max_delta.detach()
    return dataset.normalize_raw(torch.clamp(X + max_delta[:X.size(0)], min=0, max=1))

def attack_pgd_random5(model, dataset, X, y, epsilon, alpha, attack_iters, device, restarts=1, norm="l_inf", early_stop=False, fgsm_init=None, als=False, history_label = None, als_loss = None, n_sample = 5, sampled_classes=None):
    max_loss = torch.zeros(y.shape[0]).cuda(device)
    max_delta = torch.zeros_like(X).cuda(device)
    batchsize = X.shape[0]

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda(device)
        if attack_iters>1 or fgsm_init=='random':
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for iteration in range(attack_iters):
            output = model(dataset.normalize_raw(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                #early stop==True and all preds are wrong
                break
            
            new_output = torch.gather(output,1,sampled_classes)
            new_y = torch.full([batchsize], n_sample-1).cuda(device)

            # if als:
            #     loss = als_loss(new_output,new_y,history_label=history_label[new_y])
            # else:
            loss = F.cross_entropy(new_output, new_y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            # if als:
            #     all_loss = als_loss(model(dataset.normalize_raw(X+delta)), 
            #             y, history_label=history_label[])
            # else:
            
            output = model(dataset.normalize_raw(X+delta))
            new_output = torch.gather(output,1,sampled_classes)
            new_y = torch.full([batchsize], n_sample-1).cuda(device)

            all_loss = F.cross_entropy(new_output, 
                        new_y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
            max_delta = max_delta.detach()
    return dataset.normalize_raw(torch.clamp(X + max_delta[:X.size(0)], min=0, max=1))

def attack_pgd_att(model, dataset, X, y, epsilon, alpha, attack_iters, device, restarts=1, norm="l_inf", early_stop=False, fgsm_init=None, als=False, history_label = None, als_loss = None, self_att = False):
    max_loss = torch.zeros(y.shape[0]).cuda(device)
    max_delta = torch.zeros_like(X).cuda(device)
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda(device)
        if attack_iters>1 or fgsm_init=='random':
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for iteration in range(attack_iters):
            output = model(dataset.normalize_raw(X + delta), self_att = self_att)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                #early stop==True and all preds are wrong
                break
            if als:
                loss = als_loss(output,y,history_label=history_label)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            if als:
                all_loss = als_loss(model(dataset.normalize_raw(X+delta), self_att = self_att), 
                        y, history_label=history_label)
            else:
                all_loss = F.cross_entropy(model(dataset.normalize_raw(X+delta), 
                        self_att=self_att), y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
            max_delta = max_delta.detach()
    return dataset.normalize_raw(torch.clamp(X + max_delta[:X.size(0)], min=0, max=1))

def attack_pgd_fs(model, dataset, shot, X, y, epsilon, alpha, attack_iters, device, restarts=1, norm="l_inf", early_stop=False, fgsm_init=None):
    X_shape = X.shape[:-3] # 4,75
    img_shape = X.shape[-3:] # 3,32,32
    X_flat = X.view(-1,*img_shape) # 300,3,32,32
    max_loss = torch.zeros(y.shape[0]).cuda(device)
    max_delta = torch.zeros_like(X_flat).cuda(device)
    
    for _ in range(restarts):
        delta = torch.zeros_like(X_flat).cuda(device)
        if attack_iters>1 or fgsm_init=='random':
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, 0-X_flat, 1-X_flat)
        delta.requires_grad = True
        for iteration in range(attack_iters):
            output = model(shot,dataset.normalize_raw(X + delta.view(*X_shape,*img_shape)))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            n_way = output.shape[-1]
            output = output.view(-1,n_way)
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X_flat[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            output = model(shot,dataset.normalize_raw(X + delta.view(*X_shape,*img_shape)))
            output = output.view(-1,n_way)
            all_loss = F.cross_entropy(output, y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
            max_delta = max_delta.detach()
    return dataset.normalize_raw(torch.clamp(X_flat + max_delta[:X_flat.size(0)], min=0, max=1)).view(*X_shape,*img_shape)