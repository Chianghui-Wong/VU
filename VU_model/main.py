import sys
import numpy as np
import torch
from torch.autograd import Variable
import copy
import json
import yaml
import argparse
import time
import warnings
from pathlib import Path
from functools import partial
from typing import Dict, Iterable, List
import utils
import random
import time
import pickle
from utils import *

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser('Equi-Separation training script', add_help=False)

    # Data
    parser.add_argument('--dataset', default='cfm', type=str)
    parser.add_argument('--dataset_dir', default='./data/', type=str)
    parser.add_argument('--class_num', default=10, type=int)
    parser.add_argument('--color_channel', default=1, type=int)

    # Architecture
    parser.add_argument('--model', default='GFNN', type=str)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--input_size', default=100, type=int)

    # Train
    parser.add_argument('--measure', default='within_variance', type=str)
    parser.add_argument('--optimization', default='adam', type=str)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--simple_train_batch_size', default=128, type=int)
    parser.add_argument('--simple_test_batch_size', default=100, type=int)
    parser.add_argument('--epoch_num', default=600, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--ample_size_per_class', default=100, type=int)
    parser.add_argument('--normalization', default=None, type=str)
    parser.add_argument('--eps', default=None, type=str)

    # Other
    parser.add_argument('--log_dir', default='./logs/', type=str)
    parser.add_argument('--dir_path', default='./', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=str)
    parser.add_argument('--figure_dir', default='./figure/', type=str)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    return args


def main(args):
    # fix random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    checkpoint_path = Path(args.checkpoint_dir, timestamp + '.pt')
    figure_path = Path(args.figure_dir, timestamp + '.png')

    print(args)

    # dataloader
    print('loading data ...')
    with open(Path(args.dataset_dir, args.dataset, 'train.pickle'), 'rb') as f_train:
        train_data_loader = pickle.load(f_train)
    with open(Path(args.dataset_dir, args.dataset, 'test.pickle'), 'rb') as f_test:
        test_data_loader = pickle.load(f_test)

    # build model
    print('building model ...')
    # TODO add more model

    # loss function
    # loss_function = 
    
    # optimization
    if args.optimization == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train model
    print('training model ...')
    model.train()
    rate_reduction_matrix = []
    loss_list = []
    for epoch in range(args.epoch_num):

        if epoch == args.epoch_num // 3: optimizer.param_groups[0]['lr'] /= 10    
        if epoch == args.epoch_num * 2 // 3: optimizer.param_groups[0]['lr'] /= 10
   
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(train_data_loader), minibatch_size=args.simple_train_batch_size, shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(train_data_loader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(train_data_loader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            total_loss += loss * len(minibatch)
            loss.backward()
            optimizer.step()
        total_loss /= len(train_data_loader)
        print('epoch:', epoch, 'loss:', total_loss.item())
        loss_list.append(total_loss.item())

    # well trained model to verify the conjecture
    rate_reduction_list = analyze_representations(train_data_loader, copy.deepcopy(model), args)
    rate_reduction_matrix.append(rate_reduction_list)
    draw_figure(rate_reduction_matrix[-1], figure_path)

    torch.save(model.state_dict(), checkpoint_path )
    
    # load model
    model.load_state_dict(torch.load(checkpoint_path))
    train_res = simple_test_batch(train_data_loader, model, args)
    test_res = simple_test_batch(test_data_loader, model, args)
    
    print('train acc', train_res)
    print('test acc', test_res)


if __name__ == '__main__':
    opts = get_args()
    if opts.log_dir:
        Path(opts.log_dir).mkdir(parents=True, exist_ok=True)
    if opts.dataset_dir:
        Path(opts.dataset_dir).mkdir(parents=True, exist_ok=True)
    if opts.checkpoint_dir:
        Path(opts.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if opts.figure_dir:
        Path(opts.figure_dir).mkdir(parents=True, exist_ok=True)    

    # TODO add dataset generation

    main(opts)
