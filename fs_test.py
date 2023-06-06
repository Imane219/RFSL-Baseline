import argparse
import os
from sqlalchemy import true
import yaml
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import scipy.stats

import datasets
import models
import utils
import utils.few_shot as fs
from utils import mean_confidence_interval
from datasets.samplers import CategoriesSampler

torch.cuda.set_device(0)

def main(config):
    svname = args.name
    if svname is None:
        raise ValueError
    save_path = os.path.join('./save/', svname)
    if os.path.exists(save_path) == False:
        raise ValueError
    utils.set_log_path(save_path)

    ef_epoch = config.get('eval_fs_epoch')
    if ef_epoch is None:
        ef_epoch = 5

    fs_dataset = datasets.make(config['fs_dataset'],
                                **config['fs_dataset_args'])
    utils.log('\n\nfs_test: ')
    utils.log('fs dataset: {} (x{}), {}'.format(
            fs_dataset[0][0].shape, len(fs_dataset),
            fs_dataset.n_classes))
    
    n_way = config['eval_fs_args']['n_way']
    n_query = config['eval_fs_args']['n_query']
    n_shots = config['eval_fs_args']['n_shots']
    n_task = config['eval_fs_args']['n_task']
    fs_loaders = []
    for n_shot in n_shots:
        fs_sampler = CategoriesSampler(
                fs_dataset.label, n_task,
                n_way, n_shot + n_query, ep_per_batch=4)
        fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                num_workers=8, pin_memory=False)
        fs_loaders.append(fs_loader)

    if config.get('fs_attack'):
        fs_attack = True
    else:
        fs_attack = False

    base_model = models.load(torch.load(config['load'], map_location='cuda:{}'.format(args.gpu)))
    model = models.make('meta-baseline', encoder=None)
    model.encoder = base_model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        base_model = nn.DataParallel(base_model)

    #evaluation
    model.eval()
    np.random.seed(0)
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    aves_keys = []
    for n_shot in n_shots:
        aves_keys += ['fsa-' + str(n_shot)]
    if fs_attack:
        for n_shot in n_shots:
            aves_keys += ['robust_fsa-' + str(n_shot)]
    aves = {k: utils.Averager() for k in aves_keys}

    clean_lst = []
    robust_lst = []
    for n_shot in n_shots:
        clean_lst.append([])
        robust_lst.append([])

    test_epochs = args.test_epochs
    for epoch in range(1, test_epochs + 1):
        log_str = 'test epoch {}: '.format(epoch)
        for i, n_shot in enumerate(n_shots):
            for data, _ , _ in tqdm(fs_loaders[i],
                                desc='fs-' + str(n_shot), leave=False):
                x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_shot, n_query, ep_per_batch=4)
                label = fs.make_nk_label(
                            n_way, n_query, ep_per_batch=4).cuda()
                if fs_attack:
                    adv_x_query = utils.attack_pgd_fs(model, fs_dataset, 
                            x_shot, fs_dataset.convert_raw(x_query), label, 
                            config['fs_attack']['eps'] / 255., 
                            config['fs_attack']['alpha'] / 255., 
                            config['fs_attack']['iters'])
                    
                    with torch.no_grad():
                        clean_logits = model(x_shot, x_query).view(-1, n_way)
                        robust_logits = model(x_shot, adv_x_query).view(-1, n_way)
                        
                        clean_acc = utils.compute_acc(clean_logits, label)
                        robust_acc = utils.compute_acc(robust_logits, label)

                        clean_lst[i].append(clean_acc)
                        robust_lst[i].append(robust_acc)

                    aves['fsa-'+str(n_shot)].add(clean_acc)
                    aves['robust_fsa-'+str(n_shot)].add(robust_acc)
                
                else:
                    with torch.no_grad():
                        logits = model(x_shot, x_query).view(-1, n_way)
                        acc = utils.compute_acc(logits, label)
                        clean_lst[i].append(clean_acc)
                    aves['fsa-' + str(n_shot)].add(acc)
            
            if fs_attack:
                key = 'robust_fsa-{}'.format(n_shot)
                log_str += ' robust_{}: {:.2f} +- {:.2f}(%), '.format(n_shot, 
                    aves[key].item() * 100, mean_confidence_interval(robust_lst[i])*100)
                key = 'fsa-{}'.format(n_shot)
                log_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, 
                    aves[key].item() * 100, mean_confidence_interval(clean_lst[i])*100)
            else:
                key = 'fsa-{}'.format(n_shot)
                log_str += ' clean_{}: {:.2f} +- {:.2f}(%)'.format(n_shot, aves[key].item()* 100, mean_confidence_interval(clean_lst[i])*100)

        utils.log(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fs_test_cifar.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--test-epochs', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
