import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = input("Enter Excel File Name: ")
readProtocols = int(input("How many different reads did you do? (i.e OD600, Wavelength, etc. are each 1 different read.) "))
numberOfReads = int(input("How many reads were made over the time period? (Get this figure from Gen5) "))
columnsum = int(input("How many columns do you have on your plate? "))
rowsum = int(input("And how many rows? "))

dataframe1 = pd.read_excel(filename + '.xlsx')
print(dataframe1)
transposeFrame = dataframe1.values.transpose()
print(np.where(transposeFrame[0]=="OD:600")[0][0])
print(transposeFrame[3][40])
print(dataframe1.iat[np.where(transposeFrame[0]=="OD:600")[0][0],0])

#Generate Fold-Change Graph
#Fold-Change Graphs are generated using a ratio of the treated target value over proportional value over the untreated target value over proportional value.
#In the case of plate reading data, fold-change graphs should be generated using OD600 as the proportional value.

#First, get where the OD600 Data starts


od600StartIndex = np.where(transposeFrame[0]=="OD:600")[0][0] + 3
x = np.arange(0,numberOfReads).transpose()
for i in range(readProtocols-1):
    dataStartIndex = (od600StartIndex)+(numberOfReads+5)*(i+1)
    print(dataStartIndex)
    for j in range(rowsum):
        plt.figure((i+1)*j)
        for k in range(columnsum):
            y = []
            for l in range(numberOfReads):
                unTreatedOD600 = transposeFrame[3+11+(12*j),od600StartIndex+1+l]
                
                TreatedOD600 = transposeFrame[3+12*j+k,od600StartIndex+1+l]
                
                unTreatedtarget = transposeFrame[3+11+(12*j),dataStartIndex+l+1]
                
                TreatedTarget = transposeFrame[3+k+12*j,dataStartIndex+l+1]
                y.append(((TreatedTarget/TreatedOD600)/(unTreatedtarget/unTreatedOD600)))
            plt.plot(x, y, label = "line " + str(k))
        plt.legend()    
        plt.show()

    


