import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.autograd import Variable
import random
import numpy as np
from tqdm import tqdm
import logging
from cifar_model import *
from utils import data_prepare
from attack_algorithms import (corruption_laplace, corruption_gaussian, TERM, evar_risk_averse_step, pgd_origin, ERM_DataAug, CLP, ALP, KL_AE, pgd_loss, fgsm_loss, trades_loss, mart_loss, corruption_uniform, CVaR_loss, PR)
from evaluate import evaluate_aa, evaluate_PGD, evaluate_PR, evaluate_cw, cw_attack
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for Training and Evaluation")

    # General settings
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST','CIFAR10', 'CIFAR100', 'svhn', 'TinyImageNet'], help='training dataset')
    parser.add_argument('--data_root', type=str, default='./dataset/cifar_10', help='Path to dataset root')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model name (e.g., resnet18, WRN)')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--model_depth', type=int, default=50, help='Depth of the model (if applicable)')
    parser.add_argument('--model_width', type=int, default=2, help='Width of the model (if applicable)')
    parser.add_argument('--num_class', type=int, default=10, help='Number of classes')
    
    parser.add_argument('--patch', type=int, default=4, help='Number of classes')
    parser.add_argument('--cvar', type=int, default=10, help='Number of classes')


    # Training settings
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='output_model/test_res_cifar10_200', help='Path to save the model')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from a checkpoint')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval'], help='stage of running')

    # Attack settings
    parser.add_argument('--attack', type=str, default='PGD', choices=['corruption_laplace', 'corruption_gaussian', 'TERM', 'EVaR', 'pgd_origin','ERM_DataAug','ALP', 'KL', 'CLP', 'CVaR_loss_autograd', 'PGD_uniform', 'Clean', 'PGD', 'FGSM', 'PR', 'Corruption','TRADES', 'MART', 'CVaR'], help='Type of attack')
    parser.add_argument('--beta', type=float, default=6.0, help='trades balanced parameter')
    parser.add_argument('--attack_steps', type=int, default=7, help='Number of attack steps')
    parser.add_argument('--attack_eps', type=float, default=8.0, help='Attack epsilon')
    parser.add_argument('--attack_lr', type=float, default=2.0, help='Learning rate for attack')
    parser.add_argument('--random_init', action='store_true', help='Flag to enable random initialization for attack')

    args = parser.parse_args()
    return args


def load_network(checkpoint_path):
    print("Loading model from {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch'] + 1
    print("Loading Done..")
    return epoch


def save_network(save_path, epoch):
    model.eval()
    checkpoint = dict()
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    checkpoint['epoch'] = epoch
    latest = f"latest_{epoch}ep_{args.attack}.pth"
    torch.save(checkpoint, os.path.join(save_path, latest))


def accuracy(logits, target):
    _, pred = torch.max(logits, dim=1)
    correct = (pred == target).sum()
    total = target.size(0)
    acc = (float(correct) / total) * 100
    return acc


def eval_model(epoch):
    model.eval()
    acc, pgd_acc = 0, 0
    tq = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for i, (x,y) in tq:
        x, y = x.to(device), y.to(device)
        x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                         step_size=args.attack_lr/255, 
                         epsilon=args.attack_eps/255, 
                         attack_steps=args.attack_steps, 
                         attack=True)
        logits = model(x)
        pgd_logits = model(x_pgd)
        acc += accuracy(logits, y)
        pgd_acc += accuracy(pgd_logits, y)
        tq.set_description('Evaluation: clean/pgd: {:.4f}/{:.4f}'.format(acc/(i+1), pgd_acc/(i+1)))
    
    logging.info('{}:@Evaluation: clean/pgd: {:.4f}/{:.4f} lr:{}'.format(epoch, acc/(i+1), pgd_acc/(i+1), optimizer.param_groups[0]['lr']))



def train(args, save_path):
    start_epoch = 0
    print(f"start_epoch: {start_epoch} {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        acc = 0
        tq = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
        for i, (x,y) in tq:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if args.attack == 'PR':
                loss, logits = PR(model=model, x=x, y=y,
                                    step_size=args.attack_lr/255,
                                    epsilon=args.attack_eps/255,
                                    attack_steps=args.attack_steps)
               
            elif args.attack == 'Clean':
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="mean")

            elif args.attack == 'Corruption':
                loss, logits = corruption_uniform(model, x, y, epsilon=args.attack_eps/255)
   
            elif args.attack == 'ERM_DataAug':
                loss, logits = ERM_DataAug(model, x, y, epsilon=args.attack_eps/255, sample_num = 20)

            elif args.attack == 'FGSM':
                loss, logits = fgsm_loss(model, x, y, epsilon=args.attack_eps/255)

            elif args.attack == 'PGD':
                loss, logits = pgd_loss(model=model, x=x, y=y, optimizer=optimizer,
                                        step_size=args.attack_lr/255,
                                        epsilon=args.attack_eps/255,
                                        attack_steps=args.attack_steps)
                
                
            elif args.attack == "TRADES":
                loss, logits = trades_loss(model=model, x=x, y=y, optimizer=optimizer,
                                        step_size=args.attack_lr/255,
                                        epsilon=args.attack_eps/255,
                                        attack_steps=args.attack_steps,
                                        beta=args.beta)
                
            elif args.attack == "MART":
                loss, logits = mart_loss(model=model, x=x, y=y, optimizer=optimizer,
                                        step_size=args.attack_lr/255,
                                        epsilon=args.attack_eps/255,
                                        attack_steps=args.attack_steps,
                                        beta=5.0)
                
            elif args.attack == 'CVaR':
                loss, logits = CVaR_loss(model=model, x=x, y=y, optimizer=optimizer,
                                        t_step_size=1.0,
                                        epsilon=args.attack_eps/255,
                                        attack_steps=5,
                                        beta=0.5, M=args.cvar)
            
                
            elif args.attack == 'TERM':
                loss, logits = TERM(model, x, y, t=2.0)
            
            elif args.attack == 'ALP':
                loss, logits = ALP(model, x, y, optimizer, step_size=args.attack_lr/255, epsilon=args.attack_eps/255, attack_steps=10)
            
            elif args.attack == 'CLP':
                loss, logits = CLP(model, x, y, optimizer, step_size=args.attack_lr/255, epsilon=args.attack_eps/255, attack_steps=10)

 
            loss.backward()
            optimizer.step()

            acc += accuracy(logits, y)
            
            tq.set_description('Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}, lr:{}'.format(
                epoch, args.epochs, loss.item(), acc/(i+1), optimizer.param_groups[0]['lr']))

        logging.info('Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}, LR: {:.6f}'.format(
            epoch, args.epochs, loss.item(), acc / (i + 1), optimizer.param_groups[0]['lr']))
                    
        lr_scheduler.step()

        if epoch + 1 in list(range(10, args.epochs + 1, 10)):
            eval_model(epoch + 1)
            if epoch + 1 in list(range(50, args.epochs + 1, 50)):
                save_network(save_path, epoch + 1)


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    if args.model_name == 'WRN':
        model = WRN(depth=args.model_depth, width=args.model_width, num_classes=args.num_class)
    elif args.model_name == 'resnet18':
        model = resnet("resnet18", args.input_size, num_classes=args.num_class, pretrained=False)
    elif args.model_name == 'VGG':
        model = VGG('VGG19', num_classes=args.num_class)
    elif args.model_name == 'resnet34':
        model = resnet("resnet34", args.input_size, num_classes=args.num_class, pretrained=False)
    elif args.model_name == 'simpleCNN':
        INPUT_SHAPE = (1, 28, 28)
        model = MNISTNet(input_shape = INPUT_SHAPE, num_classes=args.num_class)
    elif args.model_name == 'deit_small_patch16_224':
        from vit_model_for_cifar.deit import  deit_small_patch16_224
        model = deit_small_patch16_224(pretrained = True, img_size=args.input_size, num_classes=10, patch_size=args.patch, args=args)
    elif args.model_name == "deit_tiny_patch16_224":
        from vit_model_for_cifar.deit import  deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained = True, img_size=args.input_size, num_classes=10, patch_size=args.patch, args=args)
    elif args.model_name == "vit_small_patch16_224":
        from vit_model_for_cifar.vit import  vit_small_patch16_224
        model = vit_small_patch16_224(pretrained = True, img_size=args.input_size, num_classes=10, patch_size=args.patch, args=args)
    elif args.model_name == "vit_base_patch16_224":
        from vit_model_for_cifar.vit import  vit_base_patch16_224
        model = vit_base_patch16_224(pretrained = True, img_size=args.input_size, num_classes=10, patch_size=args.patch, args=args)

    
    model.to(device)

    trainloader, testloader = data_prepare(args)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[36, 39], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    epoch = 0

    save_path = os.path.join(args.save_path, f"{args.model_name}_{args.attack}_{args.epochs}")
    log_dir = save_path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.model_name}_{args.attack}_{args.epochs}.log")
    logging.basicConfig(
        filename=log_file,
        filemode='a',  
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    param_info = f"Data root: {args.data_root}\n" + \
             f"Model name: {args.model_name}\n" + \
             f"input_size: {args.input_size}\n" + \
             f"Model depth: {args.model_depth}\n" + \
             f"Model width: {args.model_width}\n" + \
             f"Number of classes: {args.num_class}\n" + \
             f"Learning rate: {args.lr}\n" + \
             f"Batch size: {args.batch_size}\n" + \
             f"Weight decay: {args.weight_decay}\n" + \
             f"Number of epochs: {args.epochs}\n" + \
             f"Save path: {args.save_path}\n" + \
             f"Resume training: {args.resume}\n" + \
             f"Attack type: {args.attack}\n" + \
             f"Attack steps: {args.attack_steps}\n" + \
             f"Attack epsilon: {args.attack_eps}\n" + \
             f"Attack learning rate: {args.attack_lr}\n" + \
             f"cvar num: {args.cvar}\n" + \
             f"beta: {args.beta}\n"
    
    if args.resume:
        epoch = load_network(os.path.join(save_path, "latest_50ep_adv_PR.pth"))
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.phase == 'eval':
        checkpoint_path = os.path.join(log_dir, f"latest_{args.epochs}ep_{args.attack}.pth")
        epoch = load_network(checkpoint_path)
        eval_file = os.path.join(log_dir, "evaluation.txt")

        args.attack_steps = 10
        evaluate_PGD(args, model, optimizer, log_file=eval_file)

        args.attack_steps = 20
        evaluate_PGD(args, model, optimizer, log_file=eval_file)
        evaluate_cw(args, model, optimizer, log_file=eval_file)

        args.attack_steps = 100
        evaluate_PGD(args, model, optimizer, log_file=eval_file)

        evaluate_PR(model, log_file=eval_file)

        evaluate_aa(args, model, testloader, log_file=eval_file) 
    else:
        logging.info(param_info)
        print(param_info)
        train(args, save_path)
        end_time = time.time()  
        elapsed_time = int(end_time - start_time)  
        hours, remainder = divmod(elapsed_time, 3600)  
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Running time: {hours}h {minutes}mins {seconds}s")
