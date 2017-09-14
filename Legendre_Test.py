"""This script provides the testing functionality for Legendre Polynomia function"""

from numpy.polynomial.legendre import Legendre
import numpy as np
from matplotlib import pyplot as pp
from scipy.special import legendre



def test_legendre() -> None:
    """
    Plots the given legendre polynomial

    :param poly: the polynom
    :return:
    """
    k = legendre(3)
    v = k * np.array([0.42,0.1231,.35,.982])
    x = np.linspace(0.0,1.0,100)
    y = k(x)
    for i in range(6):
        print('generate polynom nr.',i)
        #pp.plot(x,y)
        a,b,c = _generate_dataset(4,100,.1)
        pp.plot(a,c)

    pp.show()

    return

def _generate_legendre_polynomial(degree:int) -> Legendre:
    """
    Generate a legendre polynom

    :param degree   the degree of the polynom
    :return:
    """
    return legendre(degree) * np.random.uniform(-1,1,size=degree+1)

def _generate_dataset_without_conceptdrift_from_legendre_polynom(degree:int,sample_size:int) -> (np.array,np.array):
    """
    Generate a dataset from a legendre polynom without conceptdrift

    :param poly: The polynom
    :return:
    """
    poly = _generate_legendre_polynomial(degree)
    x = np.random.uniform(low=-1.0,high=1.0,size=sample_size)
    return x,poly(x),poly(x)

def _add_uniform_noise(dataset:np.array,noise_level:float) -> (np.array,np.array):
    """
    Adds noise to a dataset

    :param dataset:
    :param noise_level:
    :return:
    """
    x = dataset[0]
    y = dataset[1]
    y += np.random.uniform(-1.0*noise_level,1.0*noise_level,len(y))
    y_without_noise = dataset[2]

    return x,y,y_without_noise

def _generate_dataset(degree:int,sample_size:int,noise_level:float) -> (np.array,np.array):
    """
    provide a complete dataset based on legendre polynomials

    :param degree:
    :param sample_size:
    :param noise_level:
    :return:
    """
    return _add_uniform_noise(_generate_dataset_without_conceptdrift_from_legendre_polynom(degree,sample_size),noise_level)

def data_provider(degree:list,noise_level:list,sample_size_per_subset,number_of_drifts:list) -> (np.array,np.array):
    """
    provides data points for the FIMTDD algorithm list length of degree has to be the same size as the sample size per subset, same goes for the noise level

    :param degree:
    :param noise_level:
    :return:
    """

    X = list()
    Y = list()
    O = list()
    for i in range(number_of_drifts):
        x,y,o = _generate_dataset(degree[i],sample_size_per_subset[i],noise_level[i])
        X.append(x)
        Y.append(y)
        O.append(o)
    for i in range(len(X)):
        for j in range(len(X[i])):
            yield X[i][j],Y[i][j],O[i][j]