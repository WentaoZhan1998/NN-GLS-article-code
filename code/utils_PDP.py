import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import torch
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

class PDP_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, intValue=0):
        self.intValue = intValue
    def fit(self, X, model, y=None):
        self.treshold_ = 1
        self.model = model
        return self
    def _meaning(self, x):
        return 1 if x >= self.treshold_ else 0
    def predict(self, X, y=None):
        return self.model(torch.from_numpy(X).float()).reshape(-1).detach().numpy()

def plot_PDP_realdata(model, X, names):
    Est = PDP_estimator()
    Est.fit(X, model)

    for k in range(6):
        disp = PartialDependenceDisplay.from_estimator(estimator = Est, X = X, features = [k],
                                                     feature_names = names, #0605
                                                     percentiles=(0.05,0.95))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                            wspace=0.4, hspace=0.4)
        plt.savefig("." + names[k] + ".png")


