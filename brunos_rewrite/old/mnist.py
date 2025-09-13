import numpy as np
import torch
import logging
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from ffn_minst_relu import MNISTReLU
from optimise_weights import NetLineStepProcessor, MetaData
from misc import AverageMeter
#from torch import nn
from copy import deepcopy
from trainer import Trainer, Score, StraightLineSGD

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'
#sudo rmmod nvidia_uvm
#sudo modprobe nvidia_uvm


batch_size = 60

#from torch.optim.lr_scheduler import ExponentialLR

train_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)

num_classes = len(train_data.classes)
input_dim = np.prod(train_data[0][0].shape)

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl  = DataLoader(test_data,  batch_size=batch_size)

with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.squeeze().to(device), labels.to(device)
        #logits = testNet.forward_(xx).detach().cpu()

        #prediction = logits.argmax(dim=-1).to(device)
        #loss = loss_crossentropy_np(np.transpose(logits.numpy()), labels)
        #loss_meter.update(loss)
        #accuracy_meter.update(calculate_accuracy(prediction, labels))


"""
def test_loop(testNet):
    accuracy_meter, loss_meter  = AverageMeter(), AverageMeter(),
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]
            xx = images.view(batch_size, -1)
            logits = testNet.forward_(xx).detach().cpu()

            prediction = logits.argmax(dim=-1).to(device)
            loss = loss_crossentropy_np(np.transpose(logits.numpy()), labels)
            loss_meter.update(loss)
            accuracy_meter.update(calculate_accuracy(prediction, labels))

    return accuracy_meter.avg, loss_meter.avg


net = MNISTReLU(input_dim=input_dim, input_width=1000, hidden_width=400, output_dim=num_classes)
net.set_slopes(1, 0)
loss = nn.CrossEntropyLoss(reduction='sum')

meta = MetaData(batch_size = batch_size, output_dim=num_classes, lb = 0.005, lw = 5.0, device=device)
opt = NetLineStepProcessor(net, meta, device)
opt.eta1 = 0.0001
opt.alpha = 0.2
opt.beta = 1e-5
opt.kappa_step_pp = 0.025
opt.do_logging = False

if 'compute' in locals() and compute : 
    results = [test_loop(net)]

    for _ in range(10):
        #lr for SGD
        #lr = eta_fixed if epoch < 20 else eta_fixed * 0.1 if epoch < 40 else eta_fixed * 0.01

        #mt_qq = AverageMeter()

        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)
            xx = images.view(images.shape[0], -1)

            #logging.info("####Step with Straight-line")
            #logits1 = net.forward_(xx)
            #loss1 = loss(logits1, labels)
            #opt.zero_grad()
            #loss1.backward()
            #opt.param_groups[0]['lr'] = eta_straightline(net_straightline, loss1.item())
            #opt.step()

            #logging.info("####Step with net-line 2-step version")
            step_result = opt.step(labels, xx, momentum=0.9, weight_decay=2.5e-4)

            results.append(test_loop(net))
            #mt_qq.update(step_result.qq_norm)

    #step_result = opt.step(labels, xx, momentum=0.9, weight_decay=2.5e-4)
    #mt_qq.update(step_result.qq_norm)


    #net = BasicNN(input_dim, 200, 200, num_classes).to(device)
    #t = Trainer(train_dl, test_dl, net, optimizer_cls=SGD, momentum=0.9)
    #tt = Trainer(train_dl, test_dl, net, optimizer_cls=StraightLineSGD, aa=1, momentum=0)

    """
