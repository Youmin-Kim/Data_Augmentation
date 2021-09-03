## Data_Augmentations_Pytorch

This repository intergrated various Data Augmentation methods. Our implementation is based on these repositories:

- [PyTorch Cifar Models](https://github.com/junyuseu/pytorch-cifar-models)
- [Cutout](https://github.com/uoguelph-mlrg/Cutout)
- [RandomErasing](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomErasing)
- [Mixup](https://github.com/facebookresearch/mixup-cifar10)
- [CutMix](https://github.com/clovaai/CutMix-PyTorch)
- [RICAP](https://github.com/jackryo/ricap)
- [LA (LocalAugment)](https://ieeexplore.ieee.org/document/9319662)

## Dataset
- CIFAR10, CIFAR100

## Model
- ResNet

## Training Start
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

### Training Command
- Hyperparamters for each augmentation method are fixed to same values on each original paper
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
