import matplotlib.pyplot as plt
from IPython import display
import itertools as itr
import numpy as np
import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from optimise_weights import MetaData
from optimise_weights import NetLineStepProcessor
from ffn_minst_relu import MNISTReLU
from common.util import AverageMeter, loss_crossentropy_np

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import matplotlib.pylab as plt
#from storing_results import ResultsStorageByEpochs

torch.manual_seed(134)




# #### Constants



batch_size = 64
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
slope_plus, slope_minus=1.0, 0.

epochs_per_experiment = 5 
dataset_path = "../data"


# #### MNIST dataset

image_transform = ToTensor()

train_dataset = MNIST(root=dataset_path, train=True, download=True, transform=image_transform)
test_dataset = MNIST(root=dataset_path, train=False, download=True, transform=image_transform)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
)

num_classes = len(train_dataset.classes)
loss_fn = nn.CrossEntropyLoss()


# ##### UTILS

def eta_straightline(testNet, loss_value):
    denominator = 0.0
    for param in testNet.parameters():
        denominator += ((param.grad)**2).sum().item()
    eta = loss_value/denominator
    return eta

def calculate_accuracy(prediction, target):
    return (prediction ==target).mean(dtype=float).item()


def test_loop(testNet):
    accuracy_meter, loss_meter  = AverageMeter(), AverageMeter(),
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = testNet.forward(images)
            loss = loss_fn(logits, labels)

            loss_meter.update(loss.item())
            prediction = logits.argmax(dim=-1)
            accuracy_meter.update(calculate_accuracy(prediction, labels))

    return {'Accuracy' : accuracy_meter.avg, 'Loss' : loss_meter.avg}



# #### Compare optimisers
eta_fixed = 0.01
qq_divider = 1.0
meta = MetaData(batch_size = batch_size, output_dim=num_classes, lb = 0.005, lw = 5.0, device=device)



#Straight-line
net_straightline = MNISTReLU(input_dim=28**2, input_width=1000, hidden_width=400, output_dim=num_classes)
net_straightline.set_slopes(1, 0)
net_straightline.init_weights(0, 2)
net_straightline.to(device)
loss_straightline = nn.CrossEntropyLoss(reduction='sum')
opt_straightline = optim.SGD(net_straightline.parameters(), lr=1e-2)

#Sgd scheduled lr
net_sgd = copy.deepcopy(net_straightline)
net_sgd.set_slopes(slope_plus, slope_minus)
loss_sgd = nn.CrossEntropyLoss()
opt_sgd = optim.SGD(net_sgd.parameters(), lr=1e-2, momentum=0.9, nesterov=True, weight_decay=2.5e-4)

#Net-line 2step
net_linestep = copy.deepcopy(net_straightline)
net_linestep.set_slopes(slope_plus, slope_minus)
net_linestep.train(False)
opt_linestep = NetLineStepProcessor(net_linestep, meta, device)
opt_linestep.eta1 = 0.0001
opt_linestep.alpha = 0.2
opt_linestep.beta = 1e-5
opt_linestep.kappa_step_pp = 0.025
opt_linestep.do_logging = False


if 'compute' in locals() and compute:
    results = ResultsStorageByEpochs(['straight line', 'SGD', 'netline 2-step'])
    for epoch in range(epochs_per_experiment):
        #lr for SGD
        lr = eta_fixed if epoch < 20 else eta_fixed * 0.1 if epoch < 40 else eta_fixed * 0.01

        for images, labels in train_dataloader:
            xx, labels = images.to(device), labels.to(device)

            logits1 = net_straightline.forward(xx)
            loss1 = loss_straightline(logits1, labels)
            opt_straightline.zero_grad()
            loss1.backward()
            opt_straightline.param_groups[0]['lr'] = eta_straightline(net_straightline, loss1.item())
            opt_straightline.step()

            logits2 = net_sgd.forward(xx)
            loss2 = loss_sgd(logits2, labels)
            opt_sgd.zero_grad()
            loss2.backward()
            opt_sgd.param_groups[0]['lr'] = lr
            opt_sgd.step()

            opt_linestep.step(labels, xx, momentum=0.9, weight_decay=2.5e-4)

        # test results
        results.add_epoch('straight line', test_loop(net_straightline))
        results.add_epoch('SGD', test_loop(net_sgd))
        results.add_epoch('netline 2-step', test_loop(net_linestep))

        print(results.last('Accuracy'))

    results.plot()


