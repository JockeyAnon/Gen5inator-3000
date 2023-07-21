import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mticker
from matplotlib import cm
from tsmoothie.smoother import *



filename = input("Enter Excel File Name: ")
readProtocols = int(input("How many different reads did you do? (i.e OD600, Wavelength, etc. are each 1 different read.) "))
numberOfReads = int(input("How many reads were made over the time period? (Get this figure from Gen5) "))
columnsum = int(input("How many columns do you have on your plate? "))
rowsum = int(input("And how many rows? "))
titrationBase = int(input("If titrating, on what base factor? "))
timeframe = int(input("How many points should actually be read? "))
timestart = int(input("MANUAL ADJUSTMENT: If discontinuities are observed at the start, how many points should be redacted? "))

dataframe1 = pd.read_excel(filename + '.xlsx')
print(dataframe1)
transposeFrame = dataframe1.values.transpose()
print(np.where(transposeFrame[0]=="OD:600")[0][0])
print(transposeFrame[3][40])
print(dataframe1.iat[np.where(transposeFrame[0]=="OD:600")[0][0],0])

#Generate Fold-Change Graph
#Fold-Change Graphs are generated using a ratio of the treated target value over proportional value over the untreated target value over proportional value.
#In the case of plate reading data, fold-change graphs should be generated using OD600 as the proportional value.


od600StartIndex = np.where(transposeFrame[0]=="OD:600")[0][0] + 3
for i in range(readProtocols-1):
    dataStartIndex = (od600StartIndex)+(numberOfReads+5)*(i+1)
    print(dataStartIndex)
    for j in range(rowsum):
        plt.figure((i+1)*j)
        for k in range(columnsum):
            y = []
            for l in range(numberOfReads):
                if l < timestart:
                    continue
                if l == timeframe:
                    break
                unTreatedOD600 = transposeFrame[3+11+(12*j),od600StartIndex+1+l]
                
                TreatedOD600 = transposeFrame[3+12*j+k,od600StartIndex+1+l]
                
                unTreatedtarget = transposeFrame[3+11+(12*j),dataStartIndex+l+1]
                
                TreatedTarget = transposeFrame[3+k+12*j,dataStartIndex+l+1]
                y.append(((TreatedTarget/TreatedOD600)/(unTreatedtarget/unTreatedOD600)))
                smoother = ConvolutionSmoother(window_len=80, window_type='ones')
                smoother.smooth(y)
            plt.plot(np.linspace(timestart,timeframe,timeframe-timestart), smoother.smooth_data[0], label = "Concentration: " + str((1/titrationBase)**k))
            plt.xlabel("Time (5 Min/Incr)")
            plt.ylabel("Fluorescence Intensity")
        plt.legend()    
    plt.show()   
     

#3D graph portion:
od600StartIndex = np.where(transposeFrame[0]=="OD:600")[0][0] + 3
x = np.arange(0,numberOfReads).transpose()
for i in range(readProtocols-1):
    dataStartIndex = (od600StartIndex)+(numberOfReads+5)*(i+1)
    print(dataStartIndex)
    for j in range(rowsum):
        plt.figure((i+1)*j)
        ax = plt.axes(projection='3d')
        xlist = []
        ylist = []
        zlist = []
        for k in range(columnsum):
            y = []
            for l in range(numberOfReads):
                if l < timestart:
                    continue
                if l == timeframe:
                    break
                unTreatedOD600 = transposeFrame[3+11+(12*j),od600StartIndex+1+l]
                
                TreatedOD600 = transposeFrame[3+12*j+k,od600StartIndex+1+l]
                
                unTreatedtarget = transposeFrame[3+11+(12*j),dataStartIndex+l+1]
                
                TreatedTarget = transposeFrame[3+k+12*j,dataStartIndex+l+1]
                y.append(((TreatedTarget/TreatedOD600)/(unTreatedtarget/unTreatedOD600)))
            xlist.append(np.linspace(timestart,timeframe,timeframe-timestart))
            ylist.append(np.full((timeframe-timestart, ), k))
            smoother = ConvolutionSmoother(window_len=80, window_type='ones')
            smoother.smooth(y)
            zlist.append(smoother.smooth_data[0])
        ax.set_title("3D Representation of Inducer Reaction across a Base-2 Concentration Gradient")
        ax.plot_surface(np.array(xlist), np.array(ylist), np.array(zlist),cmap=cm.coolwarm,antialiased=True)
        ax.set_xlim3d(0, timeframe-timestart)
        ax.set_ylim3d(0, columnsum-1)
        ax.set_xlabel('Time (5 min/incr)')
        ax.set_ylabel('Well Number(Concentration = 0.5 ^ Well Number)')
        ax.set_zlabel('Fluorescence Intensity') 
    plt.show()


#Unused code

# od600StartIndex = np.where(transposeFrame[0]=="OD:600")[0][0] + 3
# x = np.arange(0,numberOfReads).transpose()
# for i in range(readProtocols-1):
#     dataStartIndex = (od600StartIndex)+(numberOfReads+5)*(i+1)
#     print(dataStartIndex)
#     for j in range(rowsum):
#         plt.figure((i+1)*j)
#         ax = plt.axes(projection='3d')

#         for k in range(columnsum):
#             y = []
#             for l in range(numberOfReads):
#                 if l == timeframe:
#                     break
#                 unTreatedOD600 = transposeFrame[3+11+(12*j),od600StartIndex+1+l]
                
#                 TreatedOD600 = transposeFrame[3+12*j+k,od600StartIndex+1+l]
                
#                 unTreatedtarget = transposeFrame[3+11+(12*j),dataStartIndex+l+1]
                
#                 TreatedTarget = transposeFrame[3+k+12*j,dataStartIndex+l+1]
#                 y.append(((TreatedTarget/TreatedOD600)/(unTreatedtarget/unTreatedOD600)))
#             xdata = np.linspace(0,timeframe,timeframe)
#             ydata = np.full((timeframe, 1), k)
#             n = 15  # the larger n is, the smoother curve will be
#             b = [1.0 / n] * n
#             a = 1
#             zdata = lfilter(b, a, y)

#             ax.plot3D(xdata, ydata, zdata,)    


#         plt.show()

# od600StartIndex = np.where(transposeFrame[0]=="OD:600")[0][0] + 3
# for i in range(readProtocols-1):
#     dataStartIndex = (od600StartIndex)+(numberOfReads+5)*(i+1)
#     print(dataStartIndex)
#     for j in range(rowsum):
#         plt.figure((i+1)*j)
#         for k in range(columnsum):
#             y = []
#             for l in range(numberOfReads):
#                 if l < timestart:
#                     continue
#                 if l == timeframe:
#                     break
#                 unTreatedOD600 = transposeFrame[3+11+(12*j),od600StartIndex+1+l]
                
#                 TreatedOD600 = transposeFrame[3+12*j+k,od600StartIndex+1+l]
                
#                 unTreatedtarget = transposeFrame[3+11+(12*j),dataStartIndex+l+1]
                
#                 TreatedTarget = transposeFrame[3+k+12*j,dataStartIndex+l+1]
#                 y.append(((TreatedTarget/TreatedOD600)/(unTreatedtarget/unTreatedOD600)))
#             plt.plot(np.linspace(timestart,timeframe,timeframe-timestart), y, label = "line " + str(k))
#         plt.legend()    
#     plt.show()  