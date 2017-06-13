import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

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




if __name__ == '__main__':
    symbolDict={}
    symbolCounter=0.
    directory=os.fsencode("Flight Data")
    normalizer=Flight_Data_Normalizer()
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
                            normalizer.fit(np.array(input).reshape(1,-1))
                except(UnicodeDecodeError):
                    pass
            print( "file run through: "+str(file))
            normalizer.save()