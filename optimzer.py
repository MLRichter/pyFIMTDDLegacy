import csv
import numpy as np
from pyFIMTDD import FIMTDD as FIMTGD
from FIMTDD_LS import FIMTDD as FIMTLS
import matplotlib.pyplot as plt
import itertools
import time
from multiprocessing import Pool
import progressbar as pb
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

def abalone_test(paramlist,learner):
    cumLoss=[0]
    with open( "abalone.data", 'rt') as abalonefile:
        for row in abalonefile:
            row=row.rstrip().split(',')

            target=float(row[-1])
            if row[0]=="M":
                numgender=1.
            if row[0]=="I":
                numgender=0.5
            if row[0]=="F":
                numgender=0.
            input=[numgender]
            for item in row[1:-1]:
                input.append(float(item))

            cumLoss.append(cumLoss[-1] + np.fabs(target - learner.eval_and_learn(np.array(input), target)))
        return cumLoss[-1]

class Abalone_Optimizer(BaseEstimator):

    def __init__(self,fimtdd):
        self.fimtdd = fimtdd
        self.cumLoss = 0

    def fit(self,X,y,sample_weight=None):
        self.cumLoss = 0
        for i in range(len(X)):
            self.cumLoss += np.fabs(y-self.fimtdd.eval_and_learn(X[i],y[i]))
        return self

    def predict(self,X):
        y = list()
        for x in X:
            y.append(self.fimtdd.eval(x))
        return y

    def score(self, X,y,sample_weight=None):
        self.fit(X,y,sample_weight)
        return self.cumLoss/float(len(X))

    def get_params(self, deep=True):
        pass

    def set_params(self,**params):
        for a in params:
            if a == "gamma":
                gamma = params[a]
            if a == "learn":
                learn = params[a]
            if a == "threshold":
                threshold = params[a]
            if a == "n_min":
                n_min = params[a]
            if a == "alpha":
                alpha = params[a]
        if type(self.fimtdd) == FIMTGD:
            self.fimtdd = FIMTGD(gamma=gamma,learn=learn,n_min=n_min,threshold=threshold,alpha=alpha)
        else:
            self.fimtdd = FIMTLS(gamma=gamma,learn=learn,n_min=n_min,threshold=threshold,alpha=alpha)

def get_data():
    input = list()
    target = list()
    with open( "abalone.data", 'rt') as abalonefile:
        for row in abalonefile:
            row=row.rstrip().split(',')
            target.append(float(row[-1]))
            if row[0]=="M":
                numgender=1.
            if row[0]=="I":
                numgender=0.5
            if row[0]=="F":
                numgender=0.
            inp=[numgender]
            for item in row[1:-1]:
                inp.append(float(item))
            input.append(inp)
    return np.array(input),np.array(target)


X,y = get_data()
#GridSearchCV(Abalone_Optimizer(FIMTGD()),param_grid=param_grid,scoring=)



