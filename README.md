# RFSL-Baseline
A baseline code for adversarially Robust Few Shot Learning. We implement [AT](https://arxiv.org/abs/1706.06083) in the Few Shot Learning paradigm, with our proposed ANM method incoporated.

The code in this repo is mainly modified from [Few-shot Meta Baseline](https://github.com/yinboc/few-shot-meta-baseline), a baseline code for the FSL task.

## File structure

```
|-- RFSL-Baseline
    |-- configs
    |   |-- yamls for parameters settings.
    |-- datasets
    |   |-- py files for processing different datasets.
    |-- models
    |   |-- py files for defining different networks.
    |-- utils
    |   |-- py files used in the main python file.
    |-- train_basemodel.py : train a clean model.
    |-- train_basemodel_adversarially.py : train a robust model. (including AT and ALS-AT)
    |-- train_basemodel_adversarially_sam.py : train a robust model with sam added. (including AT-SAM and ALS-SAM)
    |-- fs_test.py : test a model with PGD attack.
```

## Requirements

This code is validated to run with Python 3.9.12, PyTorch 1.11.0, CUDA 11.3, CUDNN 8.2.0.

## Datasets

Create a folder named 'materials' and download the following datasets into the `./RFSL-Baseline/materials/` directory. Extract and rename them to 'mini-imagenet', 'cifar-fs' and 'fc100', which are consistent with the default name in the dataloaders.
- [Mini-ImageNet](https://mega.nz/file/rx0wGQyS#96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE)
- [CifarFS](https://drive.google.com/file/d/1u2QQX0vNS7V_RyAv6rS4__GVePV8-AsS/view?usp=drive_link)
- [FC100](https://drive.google.com/file/d/1Pb6uXysgD4YmNo9H-yppYdAAxv_DJFVU/view?usp=drive_link)

## Usage

Run the shell file to train or test a model.

```shell
sh run.sh
```

### 1. Clean training

The shell file should be editted as follows:
```shell
gpu=0
config='./configs/train_rn12_cifar_clean.yaml'
name='rn12-wide/cifar/clean'  # Saved path

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python train_basemodel.py --config $config --name $name --gpu $gpu
```

### 2. Adversarially training

The shell file should be editted as follows. Note that the yaml file should be editted to choose wheather to use ALS in AT.
```shell
gpu=0
config='./configs/train_rn12_cifar_at.yaml'
name='rn12-wide/cifar/at'  # Saved path

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python train_basemodel_adversarially.py --config $config --name $name --gpu $gpu
```

### 3. Adversarially training with SAM

The shell file should be editted as follows. Keep the yaml files the same. Parameters in SAM are hard coded in train_basemodel_adversarially_sam.py.
```shell
gpu=0
config='./configs/train_rn12_cifar_at.yaml'
name='rn12-wide/cifar/at_sam'  # Saved path

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python train_basemodel_adversarially_sam.py --config $config --name $name --gpu $gpu
```

### 4. Test a model

The shell file should be editted as follows. Note that the value for the key 'load' in yaml file should be editted to keep consistent with the Saved path. Testing results will be writen in the log.txt.

```shell
gpu=0
config='./configs/fs_test_cifar.yaml'
name='rn12-wide/cifar/at'  # Saved path

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python fs_test.py --config $config --name $name --gpu $gpu
```

PS: CW attack is not updated in this repo for now.

## About training params

We add some comments in `./configs/train_rn12_cifar_at.yaml` for a better understanding.

