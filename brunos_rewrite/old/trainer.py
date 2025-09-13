import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch import nn
from copy import deepcopy

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'


class BasicNN(nn.Module):
    def __init__(self, input_dim, width1, width2, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(input_dim, width1), 
                nn.ReLU(), 
                nn.Linear(width1, width2), 
                nn.ReLU(), 
                nn.Linear(width2, output_dim)
            )

    def forward(self, x):
        return self.layers(x)


class Score:
    def __init__(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.mistakes, self.loss, self.count = 0, 0, 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        loss = self.loss_fn(y_pred, y_true)
        self.loss += loss
        self.mistakes += (y_pred.argmax(dim=1) != y_true).sum().item()
        self.count += len(y_true)
        return loss

    def report(self):
        mis = f"mis = {self.mistakes:4d}/{self.count:4d}" 
        if self.count > 500 or self.count % 100 != 0 :
            mis += f" = {self.mistakes/self.count:.4f}"
        return f"loss = {self.loss:.6f}   " + mis 


class Trainer:
    def __init__(self, train_dl, test_dl, model, optimizer_cls=torch.optim.SGD, **kwargs):
        self.verbose = 1
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = deepcopy(model).to(device)
        self.initial_state_dict = model.state_dict()
        self.score = Score()

        self.scheduler = None
        self.optimizer = {
                'cls' : optimizer_cls, 
                'arg' : kwargs, 
                }
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer['obj'] = self.optimizer['cls'](self.model.parameters(), **self.optimizer['arg'])

    def reset(self):
        self.model.load_state_dict(deepcopy(self.initial_state_dict))
        self.reset_optimizer()
        if (sch := self.scheduler) is not None :
            sch['obj'] = sch['cls'](self.optimizer['obj'], **sch['kwargs'])

    def _train_loop(self):
        self.model.train()
        score = Score()
        for x, y in self.train_dl:
            x, y = x.to(device), y.to(device)
            opt = self.optimizer['obj']
            opt.zero_grad()
            y_pred = self.model(x).squeeze()
            loss = score.update(y_pred, y)
            loss.backward()
            opt.loss = loss.item()
            opt.step()
        return score.report()

    def _test_loop(self):
        self.model.eval()
        score = Score()
        with torch.no_grad():
            for x, y in self.test_dl:
                y_pred = self.model(x.to(device))
                score.update(y_pred, y.to(device))
        return score.report()

    def process_options(self, verbose=1, **optim_options):
        self.verbose=verbose
        for name, val in optim_options.items():
            self.optimizer['obj'].param_groups[0][name] = val

    def set_scheduler(self, scheduler_cls, **kwargs):
        self.scheduler = {
                'obj' : scheduler_cls(self.optimizer['obj'], **kwargs), 
                'cls' : scheduler_cls, 
                'kwargs' : kwargs
                }

    def train(self, epochs=5, verbose=1, **optim_options):
        """Warning: optim_options only modifies options from torch optimizer objects"""
        self.process_options(verbose, **optim_options)
        for i in range(epochs):
            train_res = self._train_loop()
            test_res = self._test_loop()
            if i % self.verbose == 0:
                print(f"epoch {i:3d}  train {train_res}   test {test_res}")
            if self.scheduler is not None:
                self.scheduler['obj'].step()



def straightline_lr(params, loss):
    norm_grad = sum([(param.grad ** 2).sum().item() for param in params])
    return loss/norm_grad


class StraightLineSGD(SGD):
    def __init__(self, *args, **kwargs):
        self.lr_hist = []
        if 'aa' in kwargs :
            self.aa = kwargs['aa']
            del kwargs['aa']
        super().__init__(*args, **kwargs)

    def step(self, loss=None):
        lr = self.aa * straightline_lr(self.param_groups[0]['params'], self.loss)
        self.lr_hist.append(lr)
        self.param_groups[0]['lr'] = lr 
        super().step()



if 'compute' in locals() and compute :
    from torch.optim.lr_scheduler import ExponentialLR
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import math

    train_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    num_classes = len(train_data.classes)
    input_dim = np.prod(train_data[0][0].shape)

    train_dl = DataLoader(train_data, batch_size=60)
    test_dl  = DataLoader(test_data,  batch_size=60)

    net = BasicNN(input_dim, 200, 200, num_classes).to(device)
    t = Trainer(train_dl, test_dl, net, optimizer_cls=SGD, momentum=0.9)
    tt = Trainer(train_dl, test_dl, net, optimizer_cls=StraightLineSGD, aa=1, momentum=0)
