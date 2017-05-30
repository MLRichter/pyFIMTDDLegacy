__author__='jautz'

import csv
import numpy as np
from pyFIMTDD import FIMTDD as FIMTGD
from FIMTDD_LS import FIMTDD as FIMTLS
import matplotlib.pyplot as plt
import itertools
import time



def abalone_test(paramlist,show=False):

    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=[2], threshold=paramlist[3], learn=paramlist[4])
    cumLossgd=[0]
    cumLossls=[0]
    with open( "/home/julius/FIMT/Datasets/abalone.data", 'rt') as abalonefile:
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
            #print (input, target)

            cumLossgd.append(cumLossgd[-1] + np.fabs(target - fimtgd.eval_and_learn(np.array(input), target)))
            cumLossls.append(cumLossls[-1] + np.fabs(target - fimtls.eval_and_learn(np.array(input), target)))


        if show:
            plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
            plt.plot(cumLossls[1:], label="Filter Loss")
            plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
            plt.legend()
            figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                    + "_lr"+str(paramlist[4])+".png"
            plt.savefig(figname)
            plt.hold(False)
            #plt.show()
            #time.sleep(2)
            plt.clf()
        return [cumLossgd,cumLossls]

gammalist=np.arange(0.01,0.2,0.04)
n_minlist=np.arange(10,200,20)
alphalist=np.arange(0.05,0.5,0.1)
thresholdlist=[50]
learnlist=np.arange(0.001,0.5,0.1)
minparamgd=[]
minvalgd=np.inf
minparamls=[]
minvalls=np.inf

counter=0
numberoftests=len(gammalist)*len(n_minlist)*len(alphalist)*len(thresholdlist)*len(learnlist)
for paramlist in itertools.product(gammalist, n_minlist, alphalist, thresholdlist, learnlist):
    #print (list)
    gdls=abalone_test(paramlist)
    if gdls[0][-1]<minvalgd:
        minvalgd=gdls[0][-1]
        minparamgd=paramlist
    if gdls[1][-1]<minvalls:
        minvalls=gdls[1][-1]
        minparamls=paramlist
    counter=counter+1
    print (str(counter) +" of "+ str(numberoftests)+" done.")


print("Optimal GD: "+ str(minparamgd)+ " with " + str(minvalgd))
print("Optimal LS: "+ str(minparamls)+ " with " + str(minvalls))
abalone_test(minparamgd,True)
abalone_test(minparamls,True)