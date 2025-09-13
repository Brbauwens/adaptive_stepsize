# Storing of quantities by epochs
import matplotlib.pylab as mpl

class ResultsStorageByEpochs(dict):
    """
    This class is used for storing and plotting experimental results.

    It is a dictionary of dictionaries that maps: 
    -- quantities (for example, 'train accuracy' or '2-norm gradient') and 
    -- methods (for example 'standard SGD' or 'straigth line') 
    to a list of real numbers. 
    This list represents the quantity of the method for each epoch.
    """

    def __init__(self, methods=('_'), quantities=('train_loss', 'train_acc', 'loss', 'acc')):
        super()
        for quant in quantities:
            self[quant] = {method : [] for method in methods}

    def add_epoch(self, method, quantity_dict):
        for quantity, value in quantity_dict.items():
            self[quantity][method].append(value)

    def plot(self, quantities=None):
        if quantities is None :
            quantities = self.keys()
        assert len(quantities) % 2 == 0
        fig, axes = mpl.subplots(int(len(quantities)/2), 2, figsize=(18, 6))
        axes = axes.flatten()
        colors = ['b', 'g', 'r', 'c', 'm', 'b']
        for i, quantity in enumerate(quantities):
            axes[i].set_title(quantity)
            for (method, dct), col in zip(self[quantity].items(), colors):
                axes[i].plot(self[quantity][method], color=col, alpha=.5, label=method)
            axes[i].grid()
            axes[i].legend()
        axes[-2].set_xlabel("Epochs")
        axes[-1].set_xlabel("Epochs")
        mpl.show()

