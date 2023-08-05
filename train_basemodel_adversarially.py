import argparse
import yaml
import os
from sqlalchemy import true
import scipy.stats
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
import datasets
import models
from datasets.samplers import CategoriesSampler
import utils.few_shot as fs
from utils import mean_confidence_interval
from utils.als import AdaptiveLabelSmoothingLoss
from utils.als import LabelSmoothingLoss
from utils.sam import SAM

# torch.cuda.set_device(0)

def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    # writer

    # This will cover the original config.yaml if you choose not to remove the old directory
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    ### dataset ###

    # train
    train_dataset = datasets.make(config['train_dataset'], **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, 
                num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))

    # val
    if config.get('val_dataset'):
        val = True
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
                val_dataset[0][0].shape, len(val_dataset),
                val_dataset.n_classes))
    else:
        val = False

    # test fs
    if config.get('test_fs_dataset'):
        test_fs = True
        test_ef_epoch = config.get('test_eval_fs_epoch')
        if test_ef_epoch is None:
            test_ef_epoch = 5

        test_fs_dataset = datasets.make(config['test_fs_dataset'], 
                    **config['test_fs_dataset_args'])
        utils.log('test fs dataset: {} (x{}), {}'.format(
                test_fs_dataset[0][0].shape, len(test_fs_dataset),
                test_fs_dataset.n_classes))

        n_way = config['eval_fs_args']['n_way']
        n_query = config['eval_fs_args']['n_query']
        n_shots = config['eval_fs_args']['n_shots']
        n_task = config['eval_fs_args']['n_task']
        test_fs_loaders = []
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                    test_fs_dataset.label, n_task,
                    n_way, n_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(test_fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=False)
            test_fs_loaders.append(fs_loader)
    else:
        test_fs = False

    # val fs
    if config.get('val_fs_dataset'):
        val_fs = True
        val_ef_epoch = config.get('val_eval_fs_epoch')
        if val_ef_epoch is None:
            val_ef_epoch = 5

        val_fs_dataset = datasets.make(config['val_fs_dataset'], 
                    **config['val_fs_dataset_args'])
        utils.log('val fs dataset: {} (x{}), {}'.format(
                val_fs_dataset[0][0].shape, len(val_fs_dataset),
                val_fs_dataset.n_classes))

        val_fs_loaders = []
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                    val_fs_dataset.label, n_task,
                    n_way, n_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(val_fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=False)
            val_fs_loaders.append(fs_loader)
    else:
        val_fs = False

    ### model and optimizer ###

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
        print(model)

    if test_fs or val_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if test_fs or val_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    if config.get('load'):
        optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
        optimizer.load_state_dict(model_sv['training']['optimizer_sd'])
        lr_scheduler.last_epoch = model_sv['training']['epoch']
        start_epoch = model_sv['training']['epoch']+1
    else:
        optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
        start_epoch = 1

    ### adversary ###

    if config.get('als'):
        als = True
        als_inner = config['als']['inner']
    else:
        als = False
        als_inner = False

    if config.get('ls'):
        ls = True
    else:
        ls = False

    if config.get('val_attack'):
        val_attack = True
    else:
        val_attack = False
    
    if config.get('fs_attack'):
        fs_attack = True
    else:
        fs_attack = False
    
    if als:
        if config.get('load'):
            history_label = model_sv['training']['history_label'].detach()
        else:
            history_label = np.array(train_dataset.label)
            history_label = torch.from_numpy((history_label[:, None] == np.arange(
                train_dataset.n_classes)).astype(np.float32)).cuda(device)
        als_loss = AdaptiveLabelSmoothingLoss(gamma=config['als']['gamma'], minVal=config['als']['minval'], maxVal=config['als']['maxval'])

    if ls:
        ls_loss = LabelSmoothingLoss(classes=train_dataset.n_classes, smoothing=config['ls']['ls_value'])

    ######

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(start_epoch, max_epoch + 1):
        timer_epoch.s()

        # train_loss, train_acc, val_loss, val_acc
        aves_keys = ['tl', 'ta', 'vl', 'va', 'robust_tl','robust_ta']
        if test_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]
            if fs_attack:
                for n_shot in n_shots:
                    aves_keys += ['robust_fsa-' + str(n_shot)]
        if val_fs:
            for n_shot in n_shots:
                aves_keys += ['val_fsa-' + str(n_shot)]
            if fs_attack:
                for n_shot in n_shots:
                    aves_keys += ['val_robust_fsa-' + str(n_shot)]
        if val_attack:
            aves_keys += ['robust_vl','robust_va']

        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        for data, label, idx in tqdm(train_loader, desc='train', leave=True):
            data, label ,idx = data.cuda(device), label.cuda(device), idx.cuda(device)

            adv_configs = config['adversary']
            if als_inner:
                adv_data = utils.attack_pgd(model, train_dataset, 
                        train_dataset.convert_raw(data), label, 
                        adv_configs['eps'] / 255., adv_configs['alpha'] / 255., adv_configs['iters'], device, 
                        als=als, history_label=history_label[idx], als_loss=als_loss)
            else:
                adv_data = utils.attack_pgd(model, train_dataset, 
                        train_dataset.convert_raw(data), label, 
                        adv_configs['eps'] / 255., adv_configs['alpha'] / 255., adv_configs['iters'], device)
            clean_logits = model(data)
            robust_logits = model(adv_data)

            clean_loss = F.cross_entropy(clean_logits, label)
            if als:
                if als_inner == False:
                    history_label[idx] = (config['als']['eta'] * history_label[idx] + 
                                robust_logits.softmax(-1)) / (1 + config['als']['eta'])
                robust_loss = als_loss(robust_logits,label,history_label=history_label[idx])
            elif ls:
                robust_loss = ls_loss(robust_logits,label)
            else:
                robust_loss = F.cross_entropy(robust_logits, label)

            clean_acc = utils.compute_acc(clean_logits, label)
            robust_acc = utils.compute_acc(robust_logits, label)

            optimizer.zero_grad()
            robust_loss.backward()
            optimizer.step()

            if als_inner:
                history_label[idx] = (config['als']['eta'] * history_label[idx] + 
                            robust_logits.softmax(-1)) / (1 + config['als']['eta'])

            aves['tl'].add(clean_loss.item())
            aves['robust_tl'].add(robust_loss.item())
            aves['ta'].add(clean_acc)
            aves['robust_ta'].add(robust_acc)

        # val
        if val:
            model.eval()
            for data, label, _ in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(device), label.cuda(device)

                if val_attack:
                    adv_configs = config['val_attack']
                    adv_data = utils.attack_pgd(model, val_dataset, 
                            val_dataset.convert_raw(data), label, 
                            adv_configs['eps'] / 255., adv_configs['alpha'] / 255., adv_configs['iters'], device)

                    with torch.no_grad():
                        clean_logits = model(data)
                        robust_logits = model(adv_data)

                        clean_loss = F.cross_entropy(clean_logits, label)
                        robust_loss = F.cross_entropy(robust_logits, label)
                        clean_acc = utils.compute_acc(clean_logits, label)
                        robust_acc = utils.compute_acc(robust_logits, label)

                    aves['vl'].add(clean_loss.item())
                    aves['robust_vl'].add(robust_loss.item())
                    aves['va'].add(clean_acc)
                    aves['robust_va'].add(robust_acc)

                else:
                    with torch.no_grad():
                        logits = model(data)
                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)
                    
                    aves['vl'].add(loss.item())
                    aves['va'].add(acc)

        # test fs
        if test_fs and epoch > 40 and (epoch % test_ef_epoch == 0 or epoch == max_epoch):
            fs_model.eval()
            np.random.seed(0)

            clean_lst = []
            robust_lst = []
            for n_shot in n_shots:
                clean_lst.append([])
                robust_lst.append([])

            test_fs_str = '\nfs test:'
            fs_iters = config['fs_iters']
            if fs_iters == None:
                fs_iters = 5
                
            for fs_epoch in range(1, fs_iters + 1):
                test_fs_str += '\n\tfs epoch {}: '.format(fs_epoch)
                for i, n_shot in enumerate(n_shots):
                    for data, _ , _ in tqdm(test_fs_loaders[i],
                                        desc='fs-' + str(n_shot), leave=False):
                        # shot 4,5,1,3,32,32; query 4,75,3,32,32
                        x_shot, x_query = fs.split_shot_query(
                                    data.cuda(device), n_way, n_shot, n_query, ep_per_batch=4)
                        label = fs.make_nk_label(
                                    n_way, n_query, ep_per_batch=4).cuda(device)
                        if fs_attack:
                            adv_x_query = utils.attack_pgd_fs(fs_model, test_fs_dataset, 
                                    x_shot, test_fs_dataset.convert_raw(x_query), label, 
                                    config['fs_attack']['eps'] / 255., 
                                    config['fs_attack']['alpha'] / 255., 
                                    config['fs_attack']['iters'], device)

                            with torch.no_grad():
                                clean_logits = fs_model(x_shot, x_query).view(-1, n_way)
                                robust_logits = fs_model(x_shot, adv_x_query).view(-1, n_way)
                                
                                clean_acc = utils.compute_acc(clean_logits, label)
                                robust_acc = utils.compute_acc(robust_logits, label)

                                clean_lst[i].append(clean_acc)
                                robust_lst[i].append(robust_acc)
                                    
                            aves['fsa-'+str(n_shot)].add(clean_acc)
                            aves['robust_fsa-'+str(n_shot)].add(robust_acc)

                        else:
                            with torch.no_grad():
                                logits = fs_model(x_shot, x_query).view(-1, n_way)
                                acc = utils.compute_acc(logits, label)
                                clean_lst[i].append(clean_acc)
                            aves['fsa-' + str(n_shot)].add(acc)

                    if fs_attack:
                        key = 'robust_fsa-{}'.format(n_shot)
                        test_fs_str += ' robust_{}: {:.2f} +- {:.2f}(%), '.format(n_shot, 
                            aves[key].item() * 100, mean_confidence_interval(robust_lst[i])*100)
                        key = 'fsa-{}'.format(n_shot)
                        test_fs_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, 
                            aves[key].item() * 100, mean_confidence_interval(clean_lst[i])*100)
                    else:
                        key = 'fsa-{}'.format(n_shot)
                        test_fs_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, 
                            aves[key].item()* 100, mean_confidence_interval(clean_lst[i])*100)

        # val fs
        if val_fs and (epoch % val_ef_epoch == 0 or epoch == max_epoch):
            fs_model.eval()
            np.random.seed(0)

            clean_lst = []
            robust_lst = []
            for n_shot in n_shots:
                clean_lst.append([])
                robust_lst.append([])

            val_fs_str = '\nfs val:'
            fs_iters = config['fs_iters']
            if fs_iters == None:
                fs_iters = 5
                
            for fs_epoch in range(1, fs_iters + 1):
                val_fs_str += '\n\tfs epoch {}: '.format(fs_epoch)
                for i, n_shot in enumerate(n_shots):
                    for data, _ , _ in tqdm(val_fs_loaders[i],
                                        desc='fs-' + str(n_shot), leave=False):
                        # shot 4,5,1,3,32,32; query 4,75,3,32,32
                        x_shot, x_query = fs.split_shot_query(
                                    data.cuda(device), n_way, n_shot, n_query, ep_per_batch=4)
                        label = fs.make_nk_label(
                                    n_way, n_query, ep_per_batch=4).cuda(device)
                        if fs_attack:
                            adv_x_query = utils.attack_pgd_fs(fs_model, val_fs_dataset, 
                                    x_shot, val_fs_dataset.convert_raw(x_query), label, 
                                    config['fs_attack']['eps'] / 255., 
                                    config['fs_attack']['alpha'] / 255., 
                                    config['fs_attack']['iters'], device)

                            with torch.no_grad():
                                clean_logits = fs_model(x_shot, x_query).view(-1, n_way)
                                robust_logits = fs_model(x_shot, adv_x_query).view(-1, n_way)
                                
                                clean_acc = utils.compute_acc(clean_logits, label)
                                robust_acc = utils.compute_acc(robust_logits, label)

                                clean_lst[i].append(clean_acc)
                                robust_lst[i].append(robust_acc)
                                    
                            aves['val_fsa-'+str(n_shot)].add(clean_acc)
                            aves['val_robust_fsa-'+str(n_shot)].add(robust_acc)

                        else:
                            with torch.no_grad():
                                logits = fs_model(x_shot, x_query).view(-1, n_way)
                                acc = utils.compute_acc(logits, label)
                                clean_lst[i].append(clean_acc)
                            aves['val_fsa-' + str(n_shot)].add(acc)

                    if fs_attack:
                        key = 'val_robust_fsa-{}'.format(n_shot)
                        val_fs_str += ' robust_{}: {:.2f} +- {:.2f}(%), '.format(n_shot, 
                            aves[key].item() * 100, mean_confidence_interval(robust_lst[i])*100)
                        key = 'val_fsa-{}'.format(n_shot)
                        val_fs_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, 
                            aves[key].item() * 100, mean_confidence_interval(clean_lst[i])*100)
                    else:
                        key = 'val_fsa-{}'.format(n_shot)
                        val_fs_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, aves[key].item()* 100, mean_confidence_interval(clean_lst[i])*100)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        log_str = 'epoch {}, train: robust {:.4f}|{:.4f}, clean {:.4f}|{:.4f}'.format(
                str(epoch), aves['robust_tl'], aves['robust_ta'], aves['tl'], aves['ta'])
        
        if val:
            if val_attack:
                log_str += ', val: robust {:.4f}|{:.4f}, clean {:.4f}|{:.4f}'.format(
                        aves['robust_vl'], aves['robust_va'], aves['vl'], aves['va'])
            else:
                log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
        
        if test_fs and epoch > 40 and (epoch % test_ef_epoch == 0 or epoch == max_epoch):
            log_str += test_fs_str
        if val_fs and (epoch % val_ef_epoch == 0 or epoch == max_epoch):
            log_str += val_fs_str

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        if als:
            training['history_label'] = history_label.detach()
        
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }

        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu',default='0') # string in default
    args = parser.parse_args() # a built-in Namespace

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    
    # utils.set_gpu(args.gpu)
    device = torch.device("cuda")
    main(config)