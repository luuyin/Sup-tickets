from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import re
import random
import sparselearning
import torchvision.transforms as transforms
from models import cifar_resnet, initializers, vgg

from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10
from sparselearning.resnet_cifar100 import ResNet34, ResNet18,ResNet50
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms, get_cifar100_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}

models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-10'] = (WideResNet, [28, 10, 10, 0.0])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.0])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.0])



# cyclic learning rate adjust
def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



# get model file names

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def model_files_filter(model_files,filter_itrs=["model"]):
    new_files=[]
    for filter_itr in filter_itrs:
        for  model_file in model_files:
            if filter_itr in model_file:
                new_files.append(model_file)
    return new_files



# get superpose models

def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)


def CIMA_average(decay):

    def function(cheap_tickets,rangelist,decay=decay):
        print ("superpose by CIMA ")
        print ("decay",decay)
        superpose_flag="moving_"+str(decay)
        params = {}
        pre_params = {}
        for name in cheap_tickets[0].state_dict():
            pre_params[name] = copy.deepcopy(cheap_tickets[0].state_dict()[name])

        rangelist=list(rangelist)[1:]
        for name in cheap_tickets[0].state_dict():
            for i in rangelist:
                params[name]=copy.deepcopy(cheap_tickets[i].state_dict()[name]* decay+pre_params[name]* (1 - decay))
                pre_params[name]=params[name]                    
                                            
        return params,superpose_flag
    return function


def CAA_average(cheap_tickets,rangelist,factor):
    print ("superpose by CAA using factor",factor)
    superpose_flag="CAA"
    params = {}
    for name in cheap_tickets[0].state_dict():
    #     print ("name",name)
        params[name]=copy.deepcopy(torch.sum(   torch.stack([cheap_tickets[0].state_dict()[name]* (1-factor),    cheap_tickets[1].state_dict()[name] * factor   ],dim=0).float(),dim=0))
    
    return params,superpose_flag






# save model and print
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def epochs_calculation(args):
    cyclic_epochs=args.epochs*args.cycle_rate
    if cyclic_epochs<args.cycle:
        pretrain_epoch=args.epochs
        cyclic_epochs=0
        model_n=0
    else:
        cyclic_epochs=args.epochs*args.cycle_rate
        if cyclic_epochs% args.cycle!=0:
            cyclic_epochs=cyclic_epochs-cyclic_epochs%args.cycle
        cyclic_epochs=int(cyclic_epochs)

        pretrain_epoch=args.epochs-cyclic_epochs

        model_n= cyclic_epochs//args.cycle





    print ("pretrain_epoch",pretrain_epoch)
    print ("cyclic epochs",cyclic_epochs)
    print ("creating modesl",model_n)
    print ("cycle",args.cycle)
    print ("all training epochs",args.epochs)
    print ("cycle_rate",args.cycle_rate)
    
    return pretrain_epoch,cyclic_epochs,model_n

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.final_density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

# trian and eval

def train(args, model, device, train_loader, optimizer, epoch, lr_schedule,mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    num_iters = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):

        if lr_schedule is not None:
            lr = lr_schedule(batch_idx / num_iters)
            adjust_learning_rate(optimizer, lr)
        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))


    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)






def main():

    print ("cuda counts",torch.cuda.device_count())
    print ("current ",torch.cuda.current_device())

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='vgg-like')
    parser.add_argument('--l2', type=float, default=5e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--nolrsche', action='store_true', default=False,
                        help='disable learning rate decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--indicate_method', type=str,default='Results',help='indicate_method for save path')
    
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')    
    # cyclic

    parser.add_argument('--superposing_method', type=str,default='CAA',help='indicate_method for superposing')

    parser.add_argument('--CIMA_factor', type=float,default=0.5,help='CIMA factor')


    parser.add_argument('--cycle_rate', type=float, default=0.2, 
                    help='the ratio of cycle trainig of whole training budges')

    parser.add_argument('--no_maskupdates', action='store_true', default=False,
                        help='No mask updates')
    parser.add_argument('--re_explore', action='store_true', default=False,
                        help='explore every cyclic')

    parser.add_argument('--cyc_lr', action='store_true', default=False,
                        help='whether using cyc_lr decay')
    parser.add_argument('--cycle', type=int, default=8, metavar='N',
                    help='number of epochs to train in each cycle(default: 8)')
    parser.add_argument('--lr_1', type=float, default=0.05, metavar='LR1',
                        help='maximum learning rate of cyclic RL schedual (default: 0.05)')
    parser.add_argument('--lr_2', type=float, default=0.0001, metavar='LR2',
                        help='minimum learning rate of cyclic RL schedual (default: 0.0001)')

    parser.add_argument('--pre_train', action='store_true', default=False,
                        help='whether pre_train')
    parser.add_argument('--cyclic_train', action='store_true', default=False,
                        help='whether cyclic_train')

    parser.add_argument('--update_bn', action='store_true', default=False,
                    help='whether update_bn')



    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)


    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
            outputs=10
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 100
            
        # init model
        if args.model == 'cifar_resnet_32':
            model = cifar_resnet.Model.get_model_from_name('cifar_resnet_32', initializer=initializers.kaiming_normal, outputs=outputs).to(device)
        elif args.model == 'vgg19':
            model = vgg.VGG(depth=19, dataset=args.data, batchnorm=True).to(device)
        elif args.model == 'ResNet50':
            model = ResNet50(c=outputs).to(device)
        else:
            cls, cls_args = models[args.model]
            if args.model=="vgg-like":
                cls_args[1] = outputs
            else:
                cls_args[2] = outputs
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
      

            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)

            print_and_log('=' * 60)
            print_and_log('Prune mode: {0}'.format(args.prune))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('=' * 60)

        print ("model", args.model)
        
        if args.mgpu:
            print('Using multi gpus')
            model = torch.nn.DataParallel(model).to(device)


        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')
        
        # get training epochs
        pretrain_epoch,cyclic_epochs,model_num=epochs_calculation(args)



        lr_milestones=[int(0.5*pretrain_epoch), int(0.75*pretrain_epoch)]
        
        if args.nolrsche:
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=lr_milestones, last_epoch=-1)

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()


        # reuse the model
        if args.resume:
            print_and_log("=> loading checkpoint '{}'".format(args.resume))

            model_files = os.listdir(args.resume)
            model_files=model_files_filter(model_files)
            model_files = sorted_nicely(model_files)
            print (model_files)
            checkpoint = torch.load(args.resume + str(model_files[0]))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])



        mask = None
        if args.sparse:
            decay = CosineDecay(args.prune_rate, len(train_loader) * (args.epochs))
            mask = Masking(optimizer, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=train_loader)


            mask.add_module(model, sparse_init=args.sparse_init)

        
        save_dir = "./saved_models/"+str(args.indicate_method) + '/' + str(args.model) + '/' + str(args.data) + '/cycle_num_'+str(args.cycle)+ '/pretrain_epoch_'+str(pretrain_epoch)+ '/cyclic_epochs_'+str(cyclic_epochs)  + '/density_' + str(args.final_density)   +  '/M=' + str(model_num) +   '/ex_rate=' + str(args.large_death_rate)+ '/seed' + str(args.seed)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # if not os.path.exists(save_dir): os.system("sudo mkdir -p "+str(save_dir))

        if args.pre_train:
            print ("=====================================")
            print ("begin pre training")
            best_acc=0
            for epoch in range(0, pretrain_epoch):

                t0 = time.time()

                # train the model
                cyclic_schedule=None
                train(args, model, device, train_loader, optimizer, epoch, cyclic_schedule,mask)

                if args.valid_split > 0.0:
                    val_acc = evaluate(args, model, device, valid_loader)

                if lr_scheduler: lr_scheduler.step()

                
                ## save best model

                save_best=False
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    print ("best_acc",best_acc)

                    if save_best:
                        print('Saving best pre model')
                        
                        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, filename=os.path.join(save_dir, 'premodel_best.pth'))

                print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))


            ## save last model
            print('Saving  model')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model_previous.pth' ))


        if args.no_maskupdates==True:
            mask.prune_every_k_steps=None


        if args.cyclic_train:
            print ("=====================================")
            print ("begin cyclic train")

            for epoch in range(0, cyclic_epochs):


                t0 = time.time()

                # init best acc and update the connections
                if epoch % args.cycle==0:
                    best_acc = 0.0
                    if args.re_explore:
                        print ("re_explore at every beginning of cyclic_epochs")
                        mask.truncate_weights(prune_rate=args.large_death_rate)

                if args.cyc_lr:
                    #cyclic learning rate
                    cyclic_schedule = cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
                else:
                    cyclic_schedule=None

                # train
                train(args, model, device, train_loader, optimizer, epoch,cyclic_schedule, mask)


                print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
                # valid
                if args.valid_split > 0.0:
                    val_acc = evaluate(args, model, device, valid_loader)


                # save model at lowest learning rate
                if ((epoch + 1) % args.cycle ) ==0:


                     #### save current ticket at lowest LR

                    print (" save current ticket at lowest LR ")

                    save_checkpoint({
                        'epoch': epoch ,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, filename=os.path.join(save_dir, 'model_current.pth' ))


                    print('superposing cheap tickets ')

                    
                    # get cheap tickets
                    print ("=====================================")
                    print ("get cheap tickets")
                    cheap_tickets=[]

                    model_files = os.listdir(save_dir)
                    model_files = model_files_filter(model_files)
                    model_files = sorted_nicely(model_files)
                    model_files=list(reversed(model_files))
                    print (model_files)
                    rangelist=range(len(model_files))
                    
                    for file in rangelist:   
                        
                        # init model 
                        if args.model == 'cifar_resnet_32':
                            int_model = cifar_resnet.Model.get_model_from_name('cifar_resnet_32', initializer=initializers.kaiming_normal, outputs=outputs).to(device)
                        elif args.model == 'vgg19':
                            int_model = vgg.VGG(depth=19, dataset=args.data, batchnorm=True).to(device)
                        elif args.model == 'ResNet50':
                            int_model = ResNet50(c=outputs).to(device)
                        else:
                            cls, cls_args = models[args.model]
                            if args.model=="vgg-like":
                                cls_args[1] = outputs
                            else:
                                cls_args[2] = outputs

                            int_model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
                        # print ("model", args.model)

                        if args.mgpu:
                            print('Using multi gpus')
                            int_model = torch.nn.DataParallel(model).to(device)

                        print(model_files[file])
                        # if not 'DST' in args.resume:
                        checkpoint = torch.load(save_dir + '/' +str(model_files[file]))
                        if 'state_dict' in checkpoint:
                            int_model.load_state_dict(checkpoint['state_dict'])
                        else:
                            int_model.load_state_dict(checkpoint)


                        print ("cheap ticket acc")
                        val_acc = evaluate(args, int_model, device, valid_loader)

                        cheap_tickets.append(int_model)



                    print ("set parameters of the  model")


 

                    if args.superposing_method=="CAA"  :   
                        method=CAA_average 
                        factor=   1/((epoch+ 1) // args.cycle +1)  
                        
                    elif args.superposing_method=="CIMA":
                        method =CIMA_average
                        factor=   args.CIMA_factor 

                    
                    params,superpose_flag = method(cheap_tickets,range(len(cheap_tickets)),factor)

                    # init model 
                    if args.model == 'cifar_resnet_32':
                        int_model = cifar_resnet.Model.get_model_from_name('cifar_resnet_32', initializer=initializers.kaiming_normal, outputs=outputs).to(device)
                    elif args.model == 'vgg19':
                        int_model = vgg.VGG(depth=19, dataset=args.data, batchnorm=True).to(device)
                    elif args.model == 'ResNet50':
                        int_model = ResNet50(c=outputs).to(device)
                    else:
                        cls, cls_args = models[args.model]
                        if args.model=="vgg-like":
                            cls_args[1] = outputs
                        else:
                            cls_args[2] = outputs

                        int_model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
                    # print ("model", args.model)

                    if args.mgpu:
                        print('Using multi gpus')
                        int_model = torch.nn.DataParallel(model).to(device)



                    set_model_params(int_model, params)



                    cache_mask=copy.deepcopy(mask.masks)

                    ###cal the density
                    total_size = 0
                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        total_size  += weight.numel()
                    print('Total Model parameters:', total_size)

                    sparse_size = 0
                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        sparse_size += (weight != 0).sum().int().item()
                    #     print((weight != 0.0).sum().item()/weight.numel())
                    print('Total parameters under sparsity level of {0}'.format(sparse_size / total_size))





                    ### prune
                    print ("begin prune")
                    weight_abs = []

                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        weight_abs.append(torch.abs(weight))

                    # Gather all scores in a single vector and normalise
                    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                    num_params_to_keep = int(len(all_scores) * args.final_density)

                    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                    acceptable_score = threshold[-1]


                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        cache_mask[name][:] = ((torch.abs(weight)) >= acceptable_score).float()


                    for name, tensor in int_model.named_parameters():
                        if name in cache_mask:
                            tensor.data = tensor.data * cache_mask[name]


                    ###Confirm the density
                    total_size = 0
                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        total_size  += weight.numel()
                    print('Total int_model parameters:', total_size)

                    sparse_size = 0
                    for name, weight in int_model.named_parameters():
                        if name not in cache_mask: continue
                        sparse_size += (weight != 0).sum().int().item()
                    #     print((weight != 0.0).sum().item()/weight.numel())
                    print('Total parameters under sparsity level of {0}: {1}'.format(args.final_density, sparse_size / total_size))


                    if args.update_bn:
                        print ("update_bn")
                        torch.optim.swa_utils.update_bn(train_loader, int_model,device)
                        print ("update_bn done")


                    # #evaluataed
                    val_acc = evaluate(args, int_model, device, valid_loader)



                    ## save last model
                    print('Saving superpose model')
                    save_checkpoint({
                        'epoch': "superpose",
                        'state_dict': int_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, filename=os.path.join(save_dir, 'model_previous.pth' ))
                    

                    print ("final acc",val_acc)






        print ("Congs!! ALL DONE, GOOD JOB!!")

if __name__ == '__main__':
   main()
