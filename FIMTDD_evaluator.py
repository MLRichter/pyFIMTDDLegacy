__author__='jautz'

import csv
import numpy as np

from mpl_toolkits.mplot3d import axes3d

from pyFIMTDD import FIMTDD as FIMTGD
from Greedy_FIMTDD_LS import FIMTDD as gFIMTLS
from FIMTDD_LS import FIMTDD as FIMTLS
import matplotlib.pyplot as plt
import itertools
import time
from multiprocessing import Pool
import multiprocessing as mp
import progressbar as pb
import os
import sys
from DataGenerator import *
sys.setrecursionlimit(100000)

def test2d(paramlist,show,val):
    #print(val)
    #print(paramlist)
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]
    if True:
        start = 0.0
        end = 1.0
        X = list()
        Y = list()
        x, y, z = axes3d.get_test_data(0.1)
        num_d = len(x)**2
        for i in range(len(x)):
            for j in range(len(y)):
                input = [x[i,j],y[i,j]]
                target = z[i,j]

                X.append(input)
                Y.append(target)
        data = [X,Y]
        data = np.array(data)
        data = data.transpose()
        np.random.shuffle(data)
        data = data.transpose()

        for i in range(num_d):

            input = data[0][i]
            target = data[1][i] + (np.random.uniform() - 0.5) * 0.2
            o_target = data[1][i]

            if num_d/2 < i:
                target += 1.0
                o_target += 1.0

            cumLossgd.append(cumLossgd[-1] + np.fabs(o_target - fimtgd.eval_and_learn(np.array(input), target)))
            cumLossls.append(cumLossls[-1] + np.fabs(o_target - fimtls.eval_and_learn(np.array(input), target)))
            cumLossgls.append(cumLossgls[-1] + np.fabs(o_target - gfimtls.eval_and_learn(np.array(input), target)))
            #plt.scatter(x=x,y=y)
            #plt.show()
            if show:
                f=plt.figure()
                plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
                f.hold(True)
                plt.plot(cumLossls[1:], label="Filter Loss")
               #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
                #plt.plot(avglossgd,label="Average GD Loss")
                #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
                plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
                plt.legend()
                figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                        + "_lr"+str(paramlist[4])+".png"
                plt.savefig(figname)
                #plt.show()
                f.clear()
            #print(i)
            #print(fimtgd.count_leaves())
            #print(fimtgd.count_nodes())
        return [cumLossgd,cumLossls,cumLossgls,val,paramlist]

def sine_test(paramlist,show,val):
    #print(val)
    #print(paramlist)
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]
    if True:
        start = 0.0
        end = 1.0
        x = list()
        y = list()
        for i in range(4000):

            input = np.random.uniform(0.0,1.0)*2*np.pi
            target = np.sin(input)
            if i > 2000:
                target += 1.0
            o_target = target
            noise = (np.random.uniform() - 0.5) * 0.8
            target += noise
            x.append(input)
            y.append(target)

            cumLossgd.append(cumLossgd[-1] + np.fabs(o_target - fimtgd.eval_and_learn(np.array(input), target)))
            cumLossls.append(cumLossls[-1] + np.fabs(o_target - fimtls.eval_and_learn(np.array(input), target)))
            cumLossgls.append(cumLossgls[-1] + np.fabs(o_target - gfimtls.eval_and_learn(np.array(input), target)))
        #plt.scatter(x=x,y=y)
        #plt.show()
        if show:
            f=plt.figure()
            plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
            f.hold(True)
            plt.plot(cumLossls[1:], label="Filter Loss")
           #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
            #plt.plot(avglossgd,label="Average GD Loss")
            #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
            plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
            plt.legend()
            figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                    + "_lr"+str(paramlist[4])+".png"
            plt.savefig(figname)
            #plt.show()
            f.clear()
        #print(i)
        #print(fimtgd.count_leaves())
        #print(fimtgd.count_nodes())
        return [cumLossgd,cumLossls,cumLossgls,val,paramlist]
#counter = 0

def abalone_test(paramlist,show,val):
    #print(val)
    #print(paramlist)
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]
    with open( "abalone.data", 'rt') as abalonefile:
        i = 0
        for row in abalonefile:
            i += 1
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

            cumLossgd.append(cumLossgd[-1] + np.fabs(target - fimtgd.eval_and_learn(np.array(input), target)))
            cumLossls.append(cumLossls[-1] + np.fabs(target - fimtls.eval_and_learn(np.array(input), target)))
            cumLossgls.append(cumLossgls[-1] + np.fabs(target - gfimtls.eval_and_learn(np.array(input), target)))

        if show:
            f=plt.figure()
            plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
            f.hold(True)
            plt.plot(cumLossls[1:], label="Filter Loss")
           #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
            #plt.plot(avglossgd,label="Average GD Loss")
            #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
            plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
            plt.legend()
            figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                    + "_lr"+str(paramlist[4])+".png"
            plt.savefig(figname)
            #plt.show()
            f.clear()
        #print(i)
        #print(fimtgd.count_leaves())
        #print(fimtgd.count_nodes())
        return [cumLossgd,cumLossls,cumLossgls,val,paramlist]

def flightdata_test(paramlist,show,val):
    fimtgd = FIMTGD(gamma=paramlist[0], n_min=paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls = FIMTLS(gamma=paramlist[0], n_min=paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    cumLossgd = [0]
    cumLossls = [0]
    symbolDict={}
    symbolCounter=0.
    directory=os.fsencode("Flight Data")
    for root, dirs, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root,file), 'rt') as flightdata:

                try:
                    next(flightdata)
                    for row in flightdata:
                        row=row.rstrip().split(',')

                        #input=row[0:3,5,7,8,9,11,16,17,18,23]
                        if not "NA" in row[0:4]+[row[5]]+row[7:10]+[row[11]]+row[16:19]+ [row[23]] and not "NA" in row [14]:
                            target = float(row[14])
                            input=[float(x) for x in row[0:4]+[row[5]]+[row[7]]]
                            if not row[8] in symbolDict.keys():
                                symbolDict[row[8]]=symbolCounter
                                symbolCounter=symbolCounter+1.
                            input.append(symbolDict[row[8]])
                            if not row[9] in symbolDict.keys():
                                symbolDict[row[9]]=symbolCounter
                                symboprilCounter=symbolCounter+1.
                            input.append(symbolDict[row[9]])

                            input.append(float(row[11]))
                            if not row[16] in symbolDict.keys():
                                symbolDict[row[16]]=symbolCounter
                                symbolCounter=symbolCounter+1.
                            input.append(symbolDict[row[16]])
                            if not row[17] in symbolDict.keys():
                                symbolDict[row[17]]=symbolCounter
                                symbolCounter=symbolCounter+1.
                            input.append(symbolDict[row[17]])
                            input.append(float(18))
                            input.append(float(row[23]))
                            #print ("Input: " +str(input))

                except(UnicodeDecodeError):
                    continue

                cumLossgd.append(cumLossgd[-1] + np.fabs(target - fimtgd.eval_and_learn(np.array(input), target)))
                cumLossls.append(cumLossls[-1] + np.fabs(target - fimtls.eval_and_learn(np.array(input), target)))


    if show:
        f=plt.figure()
        plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
        f.hold(True)
        plt.plot(cumLossls[1:], label="Filter Loss")
       #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
        #plt.plot(avglossgd,label="Average GD Loss")
        #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
        plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
        plt.legend()
        figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                + "_lr"+str(paramlist[4])+".png"
        plt.savefig(figname)
        #plt.show()
        f.clear()
    print(fimtgd.count_leaves(),fimtgd.count_nodes())
    return [cumLossgd,cumLossls,val,paramlist]

def line_test(paramlist,show,val):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]

    data=generate_Line(4000)

    data=np.array(sorted(data,key=lambda x:x[0]))
    o_target=data[:,-1]
    input=data[:,1:-1]
    for counter in range(len(data)):
        noise = (np.random.uniform() - 0.5) * 0.8
        target= o_target[counter] + noise
        cumLossgd.append(cumLossgd[-1] + np.fabs(o_target[counter] - fimtgd.eval_and_learn(np.array(input[counter]), target)))
        cumLossls.append(cumLossls[-1] + np.fabs(o_target[counter] - fimtls.eval_and_learn(np.array(input[counter]), target)))
        cumLossgls.append(cumLossgls[-1] + np.fabs(o_target[counter] - gfimtls.eval_and_learn(np.array(input[counter]), target)))

    if show:
        f = plt.figure()
        plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
        f.hold(True)
        plt.plot(cumLossls[1:], label="Filter Loss")
        # avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
        # plt.plot(avglossgd,label="Average GD Loss")
        # plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
        plt.title("CumLoss Ratio:" + str(min(cumLossgd[-1], cumLossls[-1]) / max(cumLossgd[-1], cumLossls[-1])))
        plt.legend()
        figname = "g" + str(paramlist[0]) + "_nmin" + str(paramlist[1]) + "_al" + str(paramlist[2]) + "_thr" + str(
            paramlist[3]) \
                  + "_lr" + str(paramlist[4]) + ".png"
        plt.savefig(figname)
        # plt.show()
        f.clear()
    return [cumLossgd, cumLossls, cumLossgls, val, paramlist]

def lexp_test(paramlist,show,val):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]

    data=generate_Lexp(4000)

    data=np.array(sorted(data,key=lambda x:x[0]))
    o_target=data[:,-1]
    input=data[:,1:-1]
    for counter in range(len(data)):
        noise = (np.random.uniform() - 0.5) * 0.8
        target= o_target[counter] + noise
        cumLossgd.append(cumLossgd[-1] + np.fabs(o_target[counter] - fimtgd.eval_and_learn(np.array(input[counter]), target)))
        cumLossls.append(cumLossls[-1] + np.fabs(o_target[counter] - fimtls.eval_and_learn(np.array(input[counter]), target)))
        cumLossgls.append(cumLossgls[-1] + np.fabs(o_target[counter] - gfimtls.eval_and_learn(np.array(input[counter]), target)))

    if show:
        f = plt.figure()
        plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
        f.hold(True)
        plt.plot(cumLossls[1:], label="Filter Loss")
        # avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
        # plt.plot(avglossgd,label="Average GD Loss")
        # plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
        plt.title("CumLoss Ratio:" + str(min(cumLossgd[-1], cumLossls[-1]) / max(cumLossgd[-1], cumLossls[-1])))
        plt.legend()
        figname = "g" + str(paramlist[0]) + "_nmin" + str(paramlist[1]) + "_al" + str(paramlist[2]) + "_thr" + str(
            paramlist[3]) \
                  + "_lr" + str(paramlist[4]) + ".png"
        plt.savefig(figname)
        # plt.show()
        f.clear()
    return [cumLossgd, cumLossls, cumLossgls, val, paramlist]

def losc_test(paramlist,show,val):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    cumLossls  =[0]
    cumLossgls =[0]

    data=generate_Losc(4000)

    data=np.array(sorted(data,key=lambda x:x[0]))
    o_target=data[:,-1]
    input=data[:,1:-1]
    for counter in range(len(data)):
        noise = (np.random.uniform() - 0.5) * 0.8
        target= o_target[counter] + noise
        cumLossgd.append(cumLossgd[-1] + np.fabs(o_target[counter] - fimtgd.eval_and_learn(np.array(input[counter]), target)))
        cumLossls.append(cumLossls[-1] + np.fabs(o_target[counter] - fimtls.eval_and_learn(np.array(input[counter]), target)))
        cumLossgls.append(cumLossgls[-1] + np.fabs(o_target[counter] - gfimtls.eval_and_learn(np.array(input[counter]), target)))

    if show:
        f = plt.figure()
        plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
        f.hold(True)
        plt.plot(cumLossls[1:], label="Filter Loss")
        # avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
        # plt.plot(avglossgd,label="Average GD Loss")
        # plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
        plt.title("CumLoss Ratio:" + str(min(cumLossgd[-1], cumLossls[-1]) / max(cumLossgd[-1], cumLossls[-1])))
        plt.legend()
        figname = "g" + str(paramlist[0]) + "_nmin" + str(paramlist[1]) + "_al" + str(paramlist[2]) + "_thr" + str(
            paramlist[3]) \
                  + "_lr" + str(paramlist[4]) + ".png"
        plt.savefig(figname)
        # plt.show()
        f.clear()
    return [cumLossgd, cumLossls, cumLossgls, val, paramlist]

def callback_func(list):
    global result_list
    global  numberoftests
    global counter
    global bar
    #print("[Thread "+str(list[2])+' ('+str(counter)+'/'+str(numberoftests)+')]: process finished')
    bar.update(counter)
    counter += 1
    result_list[list[3]] = list

def callback_err(argv=None):
    print("Error, process killed")

def find_max(result_list):
    global gammalist
    global n_minlist
    global alphalist
    global thresholdlist
    global learnlist
    global minparamgd
    global minvalgd
    global minparamls
    global minvalls
    global counter
    global numberoftests
    global minvalgls
    global minparamgls
    global c_loss_ls
    global c_loss_gd
    global c_loss_gls

    for gdls in result_list:
        if gdls[0][-1]<minvalgd:
            minvalgd=gdls[0][-1]
            minparamgd=gdls[-1]
            c_loss_gd = gdls[0]
        if gdls[1][-1]<minvalls:
            minvalls=gdls[1][-1]
            minparamls=gdls[-1]
            c_loss_ls = gdls[1]
        if gdls[2][-1]<minvalgls:
            minvalgls=gdls[2][-1]
            minparamgls=gdls[-1]
            c_loss_gls = gdls[2]
    return minvalgd,minparamgd,minvalls,minparamls,minvalgls,minparamgls

global gammalist
global n_minlist
global alphalist
global thresholdlist
global learnlist
global minparamgd
global minvalgd
global minparamls
global minvalls
global counter
global numberoftests
global result_list
global bar
global minvalgls
global minparamgls
global c_loss_ls
global c_loss_gd
global c_loss_gls

if __name__ == '__main__':

    global gammalist
    global n_minlist
    global alphalist
    global thresholdlist
    global learnlist
    global minparamgd
    global minvalgd
    global minparamls
    global minvalls
    global counter
    global numberoftests
    global result_list
    global bar
    global minvalgls
    global minparamgls
    global c_loss_ls
    global c_loss_gd
    global c_loss_gls

    #pool = #()
    if(True): #For plot testing purposes, set this to false
        #gammalist=[0.25,0.5,0.75,1.0]
        #n_minlist=[5,10,15,100]#np.arange(1,1000,50)
        #alphalist=[0.001,0.01,0.1]
        #thresholdlist= [10,25,100,150]
        #learnlist=[0.005,0.01,0.025,0.05]
        #greedlist=[2,5,50,100]

        gammalist=[0.05,0.5,1.0]
        n_minlist=[5,10,15,35]#np.arange(1,1000,50)
        alphalist=[0.001]
        thresholdlist= [5,15,100]
        learnlist=[0.01,0.025,0.05]
        greedlist=[1,5,50]
    else:
        gammalist = [1.0]
        n_minlist = [35]
        alphalist = [0.2854]
        thresholdlist = [100]
        learnlist = [0.05]
        greedlist = [2]



    minparamgd=[]
    minvalgd=np.inf
    minparamls=[]
    minvalls=np.inf
    minparamgls = []
    minvalgls = np.inf
    pool = Pool(processes=mp.cpu_count()-1)
    counter=0
    numberoftests=len(gammalist)*len(n_minlist)*len(alphalist)*len(thresholdlist)*len(learnlist)
    result_list = [None]*numberoftests
    results = np.zeros(numberoftests)
    c = 0
    bar = pb.ProgressBar(max_value=numberoftests)
    if(False): #for non pool test, set this to true
        for paramlist in itertools.product(gammalist, n_minlist, alphalist, thresholdlist, learnlist):
            paramlist = list(paramlist)
            idx = learnlist.index(paramlist[-1])
            paramlist.append(greedlist[idx])
            line_test(paramlist,False,12)
            c=c+1
    else:
        for paramlist in itertools.product(gammalist, n_minlist, alphalist, thresholdlist, learnlist):
            paramlist = list(paramlist)
            idx = learnlist.index(paramlist[-1])
            paramlist.append(greedlist[idx])
            pool.apply_async(func=sine_test,args=(paramlist,False,c),callback=callback_func)
            #callback_func(test2d(paramlist,False,c))
            c = c+1
    pool.close()
    pool.join()
    print('Proceses Finished')
    #print(result_list)
    minvalgd, minparamgd, minvalls, minparamls,minvalgls,minparamgls = find_max(result_list)
    print("Optimal GD: "+ str(minparamgd)+ " with " + str(minvalgd))
    print("Optimal LS: "+ str(minparamls)+ " with " + str(minvalls))
    print("Optimal gLS: " + str(minparamgls) + " with " + str(minvalgls))

    f=plt.figure()
    plt.plot(c_loss_gd[1:], label="FIMTGD")
    f.hold(True)
    plt.plot(c_loss_ls[1:], label="FIMTLS")
    f.hold(True)
    plt.plot(c_loss_gls[1:], label="gFIMTLS")
       #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
        #plt.plot(avglossgd,label="Average GD Loss")
        #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
    plt.legend()
    plt.savefig('optimal_performance.png')
    plt.show()
    f.clear()


    #abalone_test(minparamgd,True,0)
    #abalone_test(minparamls,True,0)