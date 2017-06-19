__author__ = "julius autz"


import numpy as np

def generate_Line(amount):
    """Generates artificial data, based on a linear function.
     It can be used to simulate concept shift, if one orders the data set by the first dimension and then discards it.
     Otherwise the data has a more complex relation.
     @:param amount the number of datapoints generated
     @:returns the artifical testdata, without noise"""

    data=[]

    for counter in range(amount):
        x1=np.random.rand()>=0.5
        x2=np.random.rand()
        x3=np.random.rand()
        if x1:
            y=1+2*x2+x3
        else:
            y=-4-2*x2-x3

        data.append([x1,x2,x3,y])
    return np.array(data)

def generate_Lexp(amount):
    """Generates artificial data, based on a euler function.
     It can be used to simulate concept shift, if one orders the data set by the first dimension and then discards it.
     Otherwise the data has a more complex relation.
     @:param amount the number of datapoints generated
     @:returns the artifical testdata, without noise"""

    data=[]
    for counter in range(amount):
        x1 = np.random.rand() >= 0.5
        x2 = np.random.rand()
        x3 = np.random.rand()
        x4 = np.random.rand()
        x5 = np.random.rand()
        if x1:
            y = 1 + 2 * x2 + 3*x3 - np.exp(-2*(x4+x5))
        else:
            y = 1 - 1-2 * x2 - 3.1*x3 + np.exp(-3*(x4-x5))
        data.append([x1,x2,x3,x4,x5,y])

    return np.array(data)

def generate_Losc(amount):
    """Generates artificial data, based on a euler+sine function.
     It can be used to simulate concept shift, if one orders the data set by the first dimension and then discards it.
     Otherwise the data has a more complex relation.
     @:param amount the number of datapoints generated
     @:returns the artifical testdata, without noise"""
    data=[]
    for counter in range(amount):
        x1 = np.random.rand() >= 0.5
        x2 = np.random.rand()
        x3 = np.random.rand()
        x4 = np.random.rand()
        x5 = np.random.rand()
        if x1:
            y = 1 + 1.5 * x2 + x3 + np.sin(2*(x4+x5))*np.exp(-2*(x2+x4))
        else:
            y = -1 - 2 * x2 - x3 + np.sin(3*(x4+x5))*np.exp(-3*(x3-x4))
        data.append([x1,x2,x3,x4,x5,y])

    return np.array(data)