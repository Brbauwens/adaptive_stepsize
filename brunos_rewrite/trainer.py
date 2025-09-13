import torch
torch.manual_seed(43)
from storing_results import ResultsStorageByEpochs 
import pickle

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'



class Score:
    def __init__(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.mistakes, self.loss, self.count = 0, 0, 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        loss = self.loss_fn(y_pred, y_true)
        self.loss += loss.item()
        self.mistakes += (y_pred.argmax(dim=1) != y_true).sum().item()
        self.count += len(y_true)
        return loss

    def loss_and_error(self):
        return self.loss/self.count, self.mistakes/self.count


class Trainer:
    """This class implements generic training in torch with learning rate shedules, 
    in which shedules can be updated in each minibatch.

    The constructor requires:
    -- train_dl, test_dl: dataloaders for training and testing
    -- model, optimizer: a torch model and optimizer
    -- scheduler: any class that has a next or next_batch function and contains a field optimizer. 
    """

    def __init__(self, train_dl, test_dl, model, optimizer, scheduler=None, results=None, method_name='_'):
        self.train_dl, self.test_dl, self.model = train_dl, test_dl, model.to(device)
        self.optimizer, self.scheduler, self.method_name = optimizer, scheduler, method_name
        if scheduler:
            if not hasattr(scheduler, 'batch_step') and not hasattr(scheduler, 'step'):
                print("Warning: scheduler has neither a 'step' nor a 'batch_step' function and is ignored.")
            assert scheduler.optimizer == optimizer, "scheduler must point to the same optimizer as the Trainer"
        self.results = ResultsStorageByEpochs() if results is None else results

    def _train_loop(self):
        self.model.train()
        score = Score()
        for x, y in self.train_dl:
            x, y = x.to(device), y.to(device)
            opt = self.optimizer
            opt.zero_grad()
            y_pred = self.model(x).squeeze()
            loss = score.update(y_pred, y)
            loss.backward()
            if hasattr(self.scheduler, 'batch_step'):
                #print(f'loss {loss.item():.4f} ', end=' ')
                self.scheduler.batch_step(loss, x, y, y_pred)
            opt.step()
        return score.loss_and_error()

    def _test_loop(self):
        self.model.eval()
        score = Score()
        with torch.no_grad():
            for x, y in self.test_dl:
                y_pred = self.model(x.to(device))
                score.update(y_pred, y.to(device))
        return score.loss_and_error()

    def _report(self, num_epoch, verbose, train_res):
        test_res = self._test_loop()
        if verbose >= 1 and num_epoch % verbose == 0:
            s = [f"{loss:.6f} {err:.4f}" for loss, err in [train_res, test_res]]
            print(f"epoch {num_epoch:3d} | train {s[0]} | test {s[1]}")
        self.results.add_epoch({
               'train_loss' : train_res[0], 'train_error' : train_res[1], 
                'test_loss' : test_res[0],   'test_error' : test_res[1]
            })
        with open("data.pkl", "wb") as file:   
            pickle.dump(self.results, file)

    def train(self, epochs=5, verbose=1):
        self.results.start_method(self.method_name)
        for num_epoch in range(epochs):
            train_res = self._train_loop()
            test_res = self._test_loop()
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            self._report(num_epoch, verbose, train_res)


if 'compute' in locals() and compute:
    from torchvision import datasets, transforms
    #from load_data import load_data
    from torch.utils.data import DataLoader
    from nets import BasicNN, simple_cnn, make_resnet9
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.optim import SGD
    from lr_schedulers import NetLine2StepScheduler, StraightLineScheduler
    #from load_data import load_data
    #from storing_results import ResultsStorageByEpochs

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

    model = make_resnet9(3, 10).to(device)
    optim = SGD(model.parameters(), momentum=0.85, weight_decay=5e-3)
    schedulerNetLine = NetLine2StepScheduler(optim, model, alpha=0.6, beta=1e-4, eta_test=5e-5)
    t = Trainer(train_dl, test_dl, model, optim, schedulerNetLine, method_name='net line')
    t.train(epochs=60)


    '''
if 'compute' in locals() and compute:
    num_epochs = 10 

    optim = SGD(model.parameters(), momentum=0, lr=0.01)
    t = Trainer(train_dl, test_dl, model, optim, scheduler=None, method_name='no scheduler')
    t.train(num_epochs)

    optim = SGD(model.parameters(), momentum=0.5)
    schedulerSL = StraightLineSheduler(optim, coeff=0.4)
    t = Trainer(train_dl, test_dl, model, optim, schedulerSL, method_name='straight line', results=t.results)
    t.train(num_epochs)
    t.reset_model()

    optim = SGD(model.parameters(), momentum=0.9)
    schedulerNetLine = NetLine2StepScheduler(model, optim, alpha=0.33)
    t = Trainer(train_dl, test_dl, model, optim, schedulerNetLine, method_name='net line 2step', results=t.results)
    t.train(num_epochs)
    t.reset()
    

    #schedulerExp = ExponentialLR(optimizer, gamma=0.96)
    #schedulerNetLine = NetLine2StepScheduler(model, optimizer, alpha=0.33)
    '''
