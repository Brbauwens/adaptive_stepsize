from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_data(dataset):
    train_data = dataset(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = dataset(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return DataLoader(train_data, shuffle=True, batch_size=60), DataLoader(test_data,  batch_size=60)
