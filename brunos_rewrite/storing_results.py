# Storing of quantities by epochs
import matplotlib.pylab as mpl
from collections import defaultdict
from time import time


class ResultsStorageByEpochs(dict):
    """
    This class is used for storing and plotting experimental results.

    It is a dictionary of dictionaries that maps: 
    -- methods (for example 'standard SGD' or 'straigth line') and
    -- quantities (for example, 'train accuracy' or '2-norm gradient')
    to a list of real numbers. 
    This list represents the quantity of the method for each epoch.

    Also, for each method, the time is stored when the experiment is finished. 
    """
    def start_method(self, method_name):
        self[method_name] = ResultsSingleExperiment()
        self.current_experiment = method_name

    def add_epoch(self, quantity_dict):
        self[self.current_experiment].add_epoch(quantity_dict)

    def _collect_quantities(self, quantities=[]):
        if len(quantities) == 0:
            for dct in self.values():
                quantities += [q for q in dct if q not in quantities]
        assert len(quantities) % 2 == 0, "TODO: implement for an odd number of quantities"
        return quantities

    def _method2color(self):
        assert len(self) <= 6, "Too many methods."
        return {method : col for method, col in zip(self.keys(), ['b', 'g', 'r', 'c', 'm', 'y'])}

    def plot(self, quantities=[], show_time=False):
        quantities = self._collect_quantities(quantities)
        fig, axes = mpl.subplots(int(len(quantities)/2), 2, figsize=(18, 6))
        axes = axes.flatten()
        m2c = self._method2color()
        for i, quantity in enumerate(quantities):
            axes[i].set_title(quantity)
            for method, exp in self.items():
                if show_time:
                    axes[i].plot(exp.runtimes, exp[quantity], color=m2c[method], alpha=.5, label=method, marker='o')
                    axes[i].set_xlim(left=0)
                    axes[i].set_xlim(right=max([max(r.runtimes) for r in self.values()]))
                else:
                    axes[i].plot(exp[quantity], color=m2c[method], alpha=.5, label=method)
            axes[i].grid()
            axes[i].legend()
        if show_time:
            axes[-2].set_xlabel("Time")
            axes[-1].set_xlabel("Time")
        else:
            axes[-2].set_xlabel("Epochs")
            axes[-1].set_xlabel("Epochs")
        mpl.show()



class ResultsSingleExperiment(defaultdict):
    def __init__(self):
        super().__init__(list)
        self.starttime = time()
        self.runtimes = []

    def add_epoch(self, quantity_dict):
        for quantity, value in quantity_dict.items():
            self[quantity].append(value)
        self.runtimes.append(time() - self.starttime)
