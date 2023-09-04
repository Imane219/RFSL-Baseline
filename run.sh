#! /bin/bash --
set -x

gpu=0
config='./configs/fs_test_fc100.yaml'
name='rn12-wide/fc100/at'


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python train_basemodel_adversarially.py --config $config --name $name --gpu $gpu

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python fs_test_cw.py --config $config --name $name --gpu $gpu