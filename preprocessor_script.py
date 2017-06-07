import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

class Flight_Data_Normalizer:

    def __init__(self, load_model = False):
        self.scaler = MinMaxScaler()
        if load_model:
            self.scaler = pickle.load(open('scaler.pkl','rb'))

    def fit(self, X,y=None):
        self.scaler.partial_fit(X,y)
        return

    def save(self):
        pickle.dump(self.scaler,open('scaler.pkl','wb+'))

    def normalize(self,X):
        return self.scaler.transform(X)