import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .utils import get_minibatches_idx
from .models import GFNN, GFNNOriginal, FNNSW, FNNWS, FNNMIX, VGG, AlexNet, GResNet, BasicBlock, Bottleneck, ResNetMixV2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def simple_test_batch(testloader, model, args):
    model.eval()
    total = 0.0
    correct = 0.0
    print('data size', len(testloader))
    minibatches_idx = get_minibatches_idx(len(testloader), minibatch_size=args.simple_test_batch_size,
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(testloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(testloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.long()).sum().item()
    test_accuracy = correct / total
    return test_accuracy

