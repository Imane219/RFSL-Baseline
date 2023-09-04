#! /bin/bash --
set -x

gpu=0
config='./configs/train_rn12_cifar_clean.yaml'
name='rn12/cifarfs/clean'


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python train_basemodel_adversarially.py --config $config --name $name --gpu $gpu

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python fs_test.py --config $config --name $name --gpu $gpu