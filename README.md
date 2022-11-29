# (UAI 2022) Sup-tickets

<div align=center><img src="https://github.com/luuyin/Sup-tickets/blob/main/sup_tickets.png" width="600" height="300"></div>


**Superposing Many Tickets into One: A Performance Booster for Sparse Neural Network Training**<br>
Lu Yin, Vlado Menkovski, Meng Fang, Tianjin Huang, Yulong Pei, Mykola Pechenizkiy, Decebal Constantin Mocanu, Shiwei Liu<br>
https://arxiv.org/abs/2205.15322<br>

Abstract: *Recent works on sparse neural network training (sparse training) have shown that a compelling trade-off between performance and efficiency can be achieved by training intrinsically sparse neural networks from scratch. Existing sparse training methods usually strive to find the best sparse subnetwork possible in one single run, without involving any expensive dense or pre-training steps. For instance, dynamic sparse training (DST), is capable of reaching a competitive performance of dense training by iteratively evolving the sparse topology during the course of training. In this paper, we argue that it is better to allocate the limited resources to create multiple low-loss sparse subnetworks and superpose them into a stronger one, instead of allocating all resources entirely to find an individual subnetwork. To achieve this, two desiderata are required: (1) efficiently producing many low-loss subnetworks, the so-called cheap tickets, within one training process limited to the standard training time used in dense training; (2) effectively superposing these cheap tickets into one stronger subnetwork. To corroborate our conjecture, we present a novel sparse training approach, termed Sup-tickets, which can satisfy the above two desiderata concurrently in a single sparse-to-sparse training process. Across various modern architectures on CIFAR-10/100 and ImageNet, we show that Sup-tickets integrates seamlessly with the existing sparse training methods and demonstrates consistent performance improvement.*


This code base is created by Lu Yin [l.yin@tue.nl](mailto:l.yin@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

This repository contains implementaions of sparse training methods: [GraNet](https://arxiv.org/abs/2106.10404), [RigL](https://arxiv.org/abs/1911.11134), [In-Time Over-Parameterization](https://arxiv.org/abs/2102.02887), [SET](https://www.nature.com/articles/s41467-018-04316-3)

The implementation is heavily based on Shiwei Liu' implemenation for [GraNet](https://github.com/VITA-Group/GraNet)

## Requirements 
The library requires Python 3.7, PyTorch v1.10.0, and CUDA v11.3.1. Other version of Pytorch should also work.

## How to Run Experiments


###  Options 

```
Options for sparse training
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --method - type of sparse training method. Choose from: GraNet, GraNet_uniform, DST, GMP, GMP_uniform
* --sparse-init - type of sparse initialization. Choose from: ERK, uniform, GMP, prune_uniform, prune_global, prune_and_grow_uniform, prune_and_grow_global, prune_structured, prune_and_grow_structured
* --model (str) - type of networks
* --growth (str) - growth mode. Choose from: random, gradient, momentum
* --prune (str) - removing mode. Choose from: magnitude, SET, threshold
* --redistribution (str) - redistribution mode. Choose from: magnitude, nonzeros, or none. (default none)
* --init-density (float) - initial density of the sparse model. (default 0.50)
* --final-density (float) - target density of the sparse model. (default 0.05)
* --init-prune-epoch (int) - the starting epoch of gradual pruning.
* --final-prune-epoch (int) - the ending epoch of gradual pruning.
* --prune-rate (float) - The pruning rate for Zero-Cost Neuroregeneration.
* --update-frequency (int) - number of training iterations between two steps of zero-cost neuroregeneration.



Options for creating and superposing cheap tickets
* --superposing_method - indicate_method for superposing
* --CIMA_factor -CIMA factor for superposing
* --cyc_lr - whether using cyc_lr decay
* --cycle - number of epochs to train in each cycle(default: 4
* --cycle_rate - the ratio of cycle trainig of whole training budges
* --lr_1 - maximum learning rate of cyclic RL schedual (default: 0.05)
* --lr_2 - minimum learning rate of cyclic RL schedual (default: 0.0001)
```

### CIFAR-10/100 Experiments
```
cd CIFAR
```

#### Superpose cheap tickets created by GraNet (s_i = 5) at sparsity 0.95:
```
python3 main_suptickets.py --indicate_method granet --update_bn --pre_train --cyclic_train --re_explore --no_maskupdates --cyc_lr --lr_2 0.005 --lr_1 0.001 --cycle 8 --cycle_rate 0.1 --sparse  --decay-schedule constant --seed 41 --sparse-init ERK --update-frequency 1000 --batch-size 128 --prune-rate 0.5 --large-death-rate 0.5 --method GraNet --growth gradient --prune magnitude --init-density 0.5 --final-density 0.05  --epochs 250  --model ResNet50 --data cifar100
```
#### Superpose cheap tickets created by Rigl at sparsity 0.95:
```
python3 main_suptickets.py --indicate_method rigl --update_bn --pre_train --cyclic_train --re_explore --no_maskupdates --cyc_lr --lr_2 0.005 --lr_1 0.001 --cycle 8 --cycle_rate 0.1 --sparse  --decay-schedule constant --seed 41 --sparse-init ERK --update-frequency 1000 --batch-size 128 --prune-rate 0.5 --large-death-rate 0.5 --method DST --growth gradient --prune magnitude --init-density 0.05 --final-density 0.05  --epochs 250  --model ResNet50 --data cifar100 
```
#### Superpose cheap tickets created by SET  at sparsity 0.95:
```
python3 main_suptickets.py --indicate_method set --update_bn --pre_train --cyclic_train --re_explore --no_maskupdates --cyc_lr --lr_2 0.005 --lr_1 0.001 --cycle 8 --cycle_rate 0.1 --sparse  --decay-schedule constant --seed 41 --sparse-init ERK --update-frequency 1000 --batch-size 128 --prune-rate 0.5 --large-death-rate 0.5 --method DST --growth random --prune magnitude --init-density 0.05 --final-density 0.05  --epochs 250  --model ResNet50 --data cifar100 
```



### Imagenet Experiments
```
cd ImageNet
```
#### Superpose cheap tickets created by GraNet (s_i = 5) at sparsity 0.90:
```
python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --first_m 30 --second_m 60 --third_m 85 --method DST --init-prune-epoch 0 --final-prune-epoch 30 --init-density 0.5  --final-density 0.1  --multiplier 1 --growth gradient --seed 17 --master_port 7768 -j20 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.1  $2 ../../imagenet2012/ --epochs 100 --indicate_method Rigl --cyclic_epochs 8 --pretrain_epoch 92 --pre_train --cyclic_train --bn_update --lr_2 0.0005 --lr_1 0.0001 --cycle 2 --large-death-rate 0.5 --cyc_lr 
```  
#### Superpose cheap tickets created by Rigl at sparsity 0.90:
```
python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --first_m 30 --second_m 60 --third_m 85 --method GraNet --init-prune-epoch 0 --final-prune-epoch 30 --init-density 0.1  --final-density 0.1  --multiplier 1 --growth gradient --seed 17 --master_port 7768 -j20 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.1  $2 ../../imagenet2012/ --epochs 100 --indicate_method Granet --cyclic_epochs 8 --pretrain_epoch 92 --pre_train --cyclic_train --bn_update --lr_2 0.0005 --lr_1 0.0001 --cycle 2 --large-death-rate 0.5 --cyc_lr 
```


# Citation

if you find this repo is helpful, please cite

```bash
@article{yin2022superposing,
  title={Superposing Many Tickets into One: A Performance Booster for Sparse Neural Network Training},
  author={Yin, Lu and Menkovski, Vlado and Fang, Meng and Huang, Tianjin and Pei, Yulong and Pechenizkiy, Mykola and Mocanu, Decebal Constantin and Liu, Shiwei},
  journal={arXiv preprint arXiv:2205.15322},
  year={2022}
}

```
