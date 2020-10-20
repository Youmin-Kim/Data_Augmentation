## Data_Augmentations_Pytorch

This repository intergrated various Data Augmentation methods. Our implementation is based on these repositories:

- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
- [Cutout](https://github.com/uoguelph-mlrg/Cutout)
- [RandomErasing](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomErasing)
- [Mixup](https://github.com/facebookresearch/mixup-cifar10)
- [CutMix](https://github.com/clovaai/CutMix-PyTorch)
- [RICAP](https://github.com/jackryo/ricap)

## Dataset
- CIFAR10, CIFAR100

## Model
- ResNet, WideResNet

## Training Start
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

### Training Command
- Hyperparamters for each augmentation methods is fixed to same values on each original paper
- ex) dataset : cifar100, augmentation : cutout, model: resnet110, index of the number of trainings: 1
```
python3 ./train.py \
--type cifar100 \
--model resnet \
--depth 110 \
--tn 1 \
--augtype cutout \
--length 8
```
- ex) dataset : cifar10, augmentation : cutout, model: wideresnet16_2, index of the number of trainings: 1
```
python3 ./train.py \
--type cifar10 \
--model wideresnet \
--depth 16 \
--wfactor 2 \
--tn 1 \
--augtype cutout \
--length 16
```
- ex) dataset : cifar100, augmentation : cutmix, model: wideresnet22_4, index of the number of trainings: 2
```
python3 ./train.py \
--type cifar100 \
--model wideresnet \
--depth 22 \
--wfactor 4 \
--tn 2 \
--augtype cutmix
```
