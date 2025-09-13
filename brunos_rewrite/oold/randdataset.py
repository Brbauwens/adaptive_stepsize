import torch
from torch.utils.data import Dataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'


class RandomDataset(Dataset):
    def __init__(self, input_dim, size, num_classes):
        self.inputs = torch.rand( (size, input_dim) )
        self.labels = torch.empty(size, dtype=torch.long).random_(num_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx,:], self.labels[idx]





if 'compute' in locals() and compute :
    from trainer import BasicNN, Trainer, StraightLineSGD
    from torch.utils.data import DataLoader
    from torch.optim import SGD

    input_dim = 5 
    num_classes = 4

    traindata = RandomDataset(input_dim, 300, num_classes)
    train_dl = DataLoader(traindata, batch_size=60)

    net = BasicNN(input_dim, 200, 200, num_classes).to(device)
    t = Trainer(train_dl, train_dl, net, optimizer_cls=SGD, momentum=0.9)
    tt = Trainer(train_dl, train_dl, net, optimizer_cls=StraightLineSGD, aa=1, momentum=0.9)
