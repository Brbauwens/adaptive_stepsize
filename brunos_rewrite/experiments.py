import torch
from torch.optim import SGD

from storing_results import ResultsStorageByEpochs 
from trainer import Trainer

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'

class Experiment:
    def __init__(self, name, quargs_SGD, scheduler, quargs_scheduler={}):
        self.name, self.quargs_SGD = name, quargs_SGD
        self.scheduler, self.quargs_scheduler =  scheduler, quargs_scheduler


def run_experiments(train_dl, test_dl, num_epochs, model, experiments):
    results = ResultsStorageByEpochs()
    initial_state = {k : v.clone() for k,v in model.state_dict().items()} 
    trainers = []
    for exp in experiments:
        print(exp.name)
        optimizer = SGD(model.parameters(), **exp.quargs_SGD)
        scheduler = None if exp.scheduler is None else exp.scheduler(optimizer, **exp.quargs_scheduler)
        t = Trainer(train_dl, test_dl, model, optimizer, scheduler, results=results, method_name=exp.name)
        t.train(epochs=num_epochs)
        trainers.append(t)
        model.load_state_dict(initial_state)
    return results, trainers


if 'compute' in locals() and compute:
    from torchvision import datasets, transforms
    #from load_data import load_data
    from torch.utils.data import DataLoader
    from nets import BasicNN, simple_cnn
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.optim import SGD
    from lr_schedulers import NetLine2StepScheduler, StraightLineScheduler
    torch.manual_seed(43)

    #train_dl, test_dl = load_data(datasets.CIFAR10)
    #model = BasicNN(train_dl, 1000, 400).to(device)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(*stats,inplace=True)])
    valid_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=valid_transforms)

    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=1024, num_workers=0, pin_memory=True)

    model = simple_cnn(3, 10).to(device)

    experiments = [
            Experiment('basic SGD', {'lr' : 0.01}, None, {}), 
            Experiment('SGD +moment +wd', {'lr' : 0.01, 'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 2.5e-4}, None, {}), 
            Experiment('Straight Line coeff=0.8', {'momentum' : 0}, StraightLineScheduler, {'coeff' : 0.8}), 
            Experiment('Straight Line coeff=0.4', {'momentum' : 0}, StraightLineScheduler, {'coeff' : 0.4}), 
            Experiment('Straight Line + moment', {'momentum' : 0.5}, StraightLineScheduler, {'coeff' : 0.3}), 
            Experiment(
                'Net Line 2step', 
                {'momentum' : 0}, 
                NetLine2StepScheduler, 
                {'model' : model, 'alpha' : 0.0078, 'eta_test' : 1e-4, 'beta' : 1e-3}
                ), 
            Experiment(
                'Net Line 2step + momentum', 
                {'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 0}, 
                NetLine2StepScheduler, 
                {'model' : model, 'alpha' : 0.5, 'beta' : 1e-3, 'eta_test' : 1e-4}
               ) 
            ]

    res, trainers = run_experiments(train_dl, test_dl, 2, model, [experiments[-1]])

   

