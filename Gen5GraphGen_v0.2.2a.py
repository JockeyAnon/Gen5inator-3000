import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mticker
from matplotlib import cm
from tsmoothie.smoother import *
import math
from scipy import signal
import matplotlib.gridspec as gridspec
import glob
from matplotlib.ticker import ScalarFormatter
import os
from matplotlib.pyplot import cm
from ast import literal_eval
import ast
import time


startTOTAL_time = time.time()
#Initial Declaration for the processed data cache. Each corresponds to a concentration, while the last one corresponds to the control. There are 12 arrays nested within each array for each graph type.
#The reason why the data is stored in cache type instead of processed on-demand is because the final data graphs use the same data but in different visualizations.

# totaldataarray = np.array([])
# od600dataarray = np.array([])
# rawnorm1array = np.array([])
# rawnorm2array = np.array([])
# raw11 = np.array([])
# rawcntrlf = np.array([])
# rawcntrlod = np.array([])
# rawcntrl = np.array([])
totaldataarray = np.array([])
od600dataarray = np.array([])
fluordataarray = np.array([])
od600dataarraywc = np.array([])
fluordataarraywc = np.array([])
controlod600 = np.array([])
controlfluor = np.array([])
rawnorm = np.array([])
rawnormc = np.array([])
fc1 = np.array([])
fc2 = np.array([])
abrstdfc1 = np.array([])
averagebyrowfc2 = np.array([])
abrstdfc2 = np.array([])
averagebysquares = np.array([])
absstd = np.array([])




#This is a customized profile based on the type of protocols done by the plate reader. This is subject to change based on the type of experiments done.
#Number of protocols done(INCLUDING OD600/absorbance readings)
readProtocols = 2
#Number of time points/individual read steps done(IMPORTANT TO GET EXACT)
numberOfReads = 217
#Number of columns on the plate
columnsum = 12
#Number of rows in the plate
rowsum = 8
#Protocol-specific: The titration factor
titrationBase = 2
#Graph-specific: The timeframe within the total number of reads the graphs should visualize
timeframe = 217
#Graph-specific: When you want to start the graphing window from the previous variable to start
timestart = 0
#Graph-specific: Which row you want to start from
figurerownumber = 0
#Graph-specific: Which column you want to start from
figurecolnumber = 0

#The list of strain names in array form. The first is formatted so sequential graph generation can take the label of the graph from this array in the same format as the graphs are referenced in subplot array format.
strainlist = [["atpB", "petA", "sucC", "rpoA", "fabA"],["A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA"],["lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]]
#This list of strain names comes in list format so the referencing of graphs is completely sequential in the case of alternative subplot referencing(This is the method used right now.)
strainlist2 = ["atpB", "petA", "sucC", "rpoA", "fabA","A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA","lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]

#Stores dataframes generated directly from pandas
#   Dataframe array format = [File index(0 = first excel sheet read)][horizontal index(all columns for each row listed in alphabetical and numerical order)][vertical index(time point)]
dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
val = "ExperimentPW"
newdir = "C:/Users/Daniel Park/Desktop/"+val+"/"
os.chdir(newdir)
path = os.getcwd() 
f = os.listdir(path)


#Checks for a "processeddata.csv" file that would have formed if the data was previously calculated. This makes it so that the program does not have to run the calculations again just to graph
# if os.path.exists(newdir+"processeddata.csv"):
#     print("Previous Calculated Data Detected!")
#     start_time = time.time()
#     #Imports the csv as a data array
#     toimport = pd.read_csv("processeddata.csv")
#     #Turns the data array into a numpy array, then transposed to align the data to the right lists
#     combinedarray = toimport.to_numpy().transpose()
#     #Distributes precalculated data into the right variables
#     totaldataarray = [literal_eval(i) for i in combinedarray[1]]
#     od600dataarray = [literal_eval(i) for i in combinedarray[2]]
#     rawnorm1array = [literal_eval(i) for i in combinedarray[3]]
#     rawnorm2array = [literal_eval(i) for i in combinedarray[4]]
#     raw11 = [literal_eval(i) for i in combinedarray[5]]
#     rawcntrlf = [literal_eval(i) for i in combinedarray[6]]
#     rawcntrlod = [literal_eval(i) for i in combinedarray[7]]
#     rawcntrl = [literal_eval(i) for i in combinedarray[8]]
#     print("--- Time elapsed to load data: %s seconds ---" % (time.time() - start_time))
#     print("Data Loaded!")
#Checks for a "processeddata.csv" file that would have formed if the data was previously calculated. This makes it so that the program does not have to run the calculations again just to graph
# if os.path.exists(newdir+"processeddata.csv"):
#     print("Previous Calculated Data Detected!")
#     start_time = time.time()
#     #Imports the csv as a data array
#     toimport = pd.read_csv("processeddata.csv").to_numpy().transpose()
#     #Turns the data array into a numpy array, then transposed to align the data to the right lists
#     print(literal_eval(toimport[1][0]))
#     #Distributes precalculated data into the right variables
#     od600dataarray = [i.replace('\n', '') for i in toimport.loc[:,"od600"].to_numpy()]
#     fluordataarray = [literal_eval(i) for i in combinedarray[2]]
#     od600dataarraywc = [literal_eval(i) for i in combinedarray[3]]
#     fluordataarraywc = [literal_eval(i) for i in combinedarray[4]]
#     controlod600 = [literal_eval(i) for i in combinedarray[5]]
#     controlfluor = [literal_eval(i) for i in combinedarray[6]]
#     rawnorm = [literal_eval(i) for i in combinedarray[7]]
#     rawnormc = [literal_eval(i) for i in combinedarray[8]]
#     fc1 = [literal_eval(i) for i in combinedarray[9]]
#     fc2 = [literal_eval(i) for i in combinedarray[10]]
#     averagebyrowfc1 = [literal_eval(i) for i in combinedarray[11]]
#     abrstdfc1 = [literal_eval(i) for i in combinedarray[12]]
#     averagebyrowfc2 = [literal_eval(i) for i in combinedarray[13]]
#     abrstdfc2 = [literal_eval(i) for i in combinedarray[14]]
#     averagebysquares = [literal_eval(i) for i in combinedarray[15]]
#     absstd = [literal_eval(i) for i in combinedarray[16]]
#     print("--- Time elapsed to load data: %s seconds ---" % (time.time() - start_time))
#     print("Data Loaded!")
# else:
#     print("No Previous Calculated Data Detected!")
#     start_time = time.time()
#     csv_files = glob.glob(os.path.join(path, "*.xlsx")) 

#     #Turns all the xlsx spreadsheets into dataframes and transposed to make index calls easier
#     for f in csv_files:
#         dataframes.append(pd.read_excel(f).values.transpose())
#     #Reads the dataframe to know where the OD600 starts
#     od600StartIndex = np.where(dataframes[0][0]=="OD:600")[0][0] + 3

#     #fold change values are calculated as follows (data/od600)/(control data/control od600)
#     def foldchangeval(i,k,l,m,n):
#         #ex1. dataframes[i][3+12*k+l][n+m] = referencing on sheet i the value in the cell 4(3+1) + l(the concentration in particular) + 12*k(row number) columns from the left and n+m(dataframe start value + read number)
#         #ex2. dataframes[i][3+12*k+11][n+m] = referencing on sheet i the value in the cell 4(3+1) + 11(drags it to the final column value(control)) + 12*k(row number) columns from the left and n+m(dataframe start value + read number) 
#         return ((dataframes[i][3+l+12*k][n+m]/dataframes[i][3+l+12*k][od600StartIndex+m])/(dataframes[i][3+11+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m]))
#     #Raw Fluorescense data from non control normalized by the corresponding instance's od600
#     def rawnorm1(i,k,l,m,n):
#         return ((dataframes[i][3+l+(12*k)][n+m]/dataframes[i][3+12*k+l][od600StartIndex+m])-(dataframes[i][3+11+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m]))
#     #Raw Fluorescense data from control normalized by the corresponding instance's od600
#     def rawnorm2(i,k,l,m,n):
#         return ((dataframes[i][3+l+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m]))


#     #This is for the fold change magnitudes

#     #Goes through the dataframes
#     for i in range(len(dataframes)):

#         for j in range(readProtocols-1):
#             n = (od600StartIndex)+(numberOfReads+4)*(j+1)
#             for k in range(rowsum):
#                 for l in range(columnsum):
#                     y = []
#                     od = []
#                     raw1 = []
#                     raw2 = []
#                     raw = []
#                     rawf = []
#                     rawod=[]
#                     rawcomb=[]
#                     for m in range(numberOfReads):
#                         if m < timestart:
#                             continue
#                         if m == timeframe:
#                             break
#                         y.append(foldchangeval(i,k,l,m,n))
#                         od.append(dataframes[i][3+12*k+l][od600StartIndex+m])
#                         raw1.append(rawnorm1(i,k,l,m,n))
#                         raw2.append(rawnorm2(i,k,l,m,n))
#                         raw.append(dataframes[i][3+11+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m])
#                         rawf.append(dataframes[i][3+12*k+l][n+m])
#                         rawod.append(dataframes[i][3+12*k+l][od600StartIndex+m])
#                         rawcomb.append(dataframes[i][3+12*k+l][n+m]/dataframes[i][3+12*k+l][od600StartIndex+m])
#                     totaldataarray[l].append(y)
#                     od600dataarray[l].append(od)
#                     rawnorm1array[l].append(raw1)
#                     rawnorm2array[l].append(raw2)
#                     raw11[l].append(raw)
#                     rawcntrlf[l].append(rawf)
#                     rawcntrlod[l].append(rawod)
#                     rawcntrl[l].append(rawcomb)
#     dict = {'total': totaldataarray, 'od600': od600dataarray, 'raw1': rawnorm1array, 'raw2': rawnorm2array, 'raw': raw11, 'rawf': rawcntrlf, 'rawod': rawcntrlod, 'rawcomb': rawcntrl}
#     df = pd.DataFrame(dict)
#     df.to_csv('processeddata.csv')
#     print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - start_time))
#     print('Data file generated!')

print("No Previous Calculated Data Detected!")
start_time = time.time()
csv_files = glob.glob(os.path.join(path, "*.xlsx")) 

#Turns all the xlsx spreadsheets into dataframes and transposed to make index calls easier
for f in csv_files:
    dataframes.append(pd.read_excel(f).values.transpose())

od600dataarraywc = np.array_split(np.append(od600dataarray,dataframes[0]),96)
fluordataarraywc = np.array_split(np.append(fluordataarray,dataframes[1]),96)
controlod600 = od600dataarraywc[12 - 1::12]
controlfluor = fluordataarraywc[12 - 1::12]
od600dataarray = np.delete(od600dataarraywc, np.s_[0::12], 0)
fluordataarray = np.delete(fluordataarraywc, np.s_[0::12], 0)
rawnorm = np.divide(fluordataarray,od600dataarray)
rawnormwc = np.divide(fluordataarraywc,od600dataarraywc)
rawnormc = np.divide(controlfluor,controlod600)
c=0
for i in range(8):
    for j in range(12):
        fc1 = np.append(fc1,np.divide(rawnormwc[c],rawnormc[i]))
        c+=1

fc1 = np.array_split(np.array_split(fc1,96),8)
fc2 = [np.delete(i, 11, 0) for i in fc1]
averagebyrowfc1 = [np.mean(i,axis=0) for i in np.array_split(fc1, 4)]
abrstdfc1 = [np.std(i,axis=0) for i in np.array_split(fc1, 4)]
averagebyrowfc2 = [np.mean(i,axis=0) for i in np.array_split(fc2, 4)]
abrstdfc2 = [np.std(i,axis=0) for i in np.array_split(fc2, 4)]
averagebysquares = [[np.mean(j,axis=0) for j in np.array_split(i, 6)] for i in averagebyrowfc1]
absstd = [[np.std(j,axis=0) for j in np.array_split(i, 6)] for i in averagebyrowfc1]
    

print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - start_time))


newpath1 = newdir + "Graphics/"     
if not os.path.exists(newpath1):
    os.makedirs(newpath1)
length = len(fc1[0][0])

# This is for first run, fixed concentration, time dependent
print("Generating Graph Set 1...")
start_time = time.time()
for i in range(8):
    for j in range(12):
        tempfig = plt.figure()
        plt.plot(np.linspace(0,length,length), fc1[i][j], '.', markersize = 3,color = 'blue')
        newpath = newpath1 + "Fixed Concentration, Time/Singles/WITH CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
        tempfig.savefig(filename)
        plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 2...")
start_time = time.time()
for i in range(8):
    for j in range(11):
        tempfig = plt.figure()
        plt.plot(np.linspace(0,length,length), fc2[i][j], '.', markersize = 3,color = 'blue')
        newpath = newpath1 + "Fixed Concentration, Time/Singles/NO CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
        tempfig.savefig(filename)
        plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 3...")
color = cm.rainbow(np.linspace(0, 1, 12))
start_time = time.time()
for i in range(8):
    tempfig = plt.figure()
    for j , c in enumerate(color):
        plt.plot(np.linspace(0,length,length), fc1[i][j], '.', markersize = 3,color = c)
        newpath = newpath1 + "Fixed Concentration, Time/stacked/WITH CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 4...")
start_time = time.time()
color = cm.rainbow(np.linspace(0, 1, 11))
for i in range(8):
    tempfig = plt.figure()
    for j, c in enumerate(color):
        plt.plot(np.linspace(0,length,length), fc2[i][j], '.', markersize = 3,color = c)
        newpath = newpath1 + "Fixed Concentration, Time/stacked/NO CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 5...")
color = cm.rainbow(np.linspace(0, 1, 12))
start_time = time.time()
for i in range(8):
    tempfig = plt.figure()
    for j , c in enumerate(color):
        plt.plot(np.linspace(0,length,length), signal.savgol_filter(fc1[i][j], 50, 1), '-', markersize = 3,color = c)
        newpath = newpath1 + "Fixed Concentration, Time/stacked/WITH CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 6...")
start_time = time.time()
color = cm.rainbow(np.linspace(0, 1, 11))
for i in range(8):
    tempfig = plt.figure()
    for j, c in enumerate(color):
        plt.plot(np.linspace(0,length,length), signal.savgol_filter(fc1[i][j], 50, 1), '-', markersize = 3,color = c)
        newpath = newpath1 + "Fixed Concentration, Time/stacked/NO CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 7...")
color = cm.rainbow(np.linspace(0, 1, 12))
start_time = time.time()
for i in range(4):
    tempfig = plt.figure()
    for j , c in enumerate(color):
        plt.plot(np.linspace(0,length,length), signal.savgol_filter(averagebyrowfc1[i][j], 50, 1), '-', markersize = 3,color = c)
        highs = signal.savgol_filter(np.add(averagebyrowfc1[i][j],abrstdfc1[i][j]), 50, 1)
        lows = signal.savgol_filter(np.subtract(averagebyrowfc1[i][j],abrstdfc1[i][j]), 50, 1)
        plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
        newpath = newpath1 + "Fixed Concentration, Time/stackedavg/WITH CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 8...")
start_time = time.time()
color = cm.rainbow(np.linspace(0, 1, 11))
for i in range(4):
    tempfig = plt.figure()
    for j, c in enumerate(color):
        plt.plot(np.linspace(0,length,length), signal.savgol_filter(averagebyrowfc2[i][j], 50, 1), '-', markersize = 3,color = c)
        highs = signal.savgol_filter(np.add(averagebyrowfc2[i][j],abrstdfc2[i][j]), 50, 1)
        lows = signal.savgol_filter(np.subtract(averagebyrowfc2[i][j],abrstdfc2[i][j]), 50, 1)
        plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
        newpath = newpath1 + "Fixed Concentration, Time/stackedavg/NO CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")

print("Generating Graph Set 9...")
color = cm.rainbow(np.linspace(0, 1, 6))
start_time = time.time()
for i in range(4):
    tempfig = plt.figure()
    for j , c in enumerate(color):
        plt.plot(np.linspace(0,length,length), signal.savgol_filter(averagebysquares[i][j], 50, 1), '-', markersize = 3,color = c)
        highs = signal.savgol_filter(np.add(averagebysquares[i][j],absstd[i][j]), 50, 1)
        lows = signal.savgol_filter(np.subtract(averagebysquares[i][j],absstd[i][j]), 50, 1)
        plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
        newpath = newpath1 + "Fixed Concentration, Time/stackedavgsqr/WITH CONTROL/" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)
print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
print("Graphs Generated and Stored!")




# # This is for first run, fixed concentration, time dependent
# print("Generating Graph Set 1...")
# start_time = time.time()
# for i in range(12):
#     for j in range(15):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         hold.plot(np.linspace(0,217,217), totaldataarray[i][0+2*j], '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "First Runs/Fixed Concentration, Time/Singles/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'first runs concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# #This is for the second run, fixed concentration, time dependent
# print("Generating Graph Set 2...")
# start_time = time.time()
# for i in range(12):
#     for j in range(15):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         hold.plot(np.linspace(0,217,217), totaldataarray[i][1+2*j], '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Second Runs/Fixed Concentration, Time/Singles/"  
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'second runs concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for first run, fixed time, concentration dependent
# print("Generating Graph Set 3...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         concentration = []
#         for k in range(12):
#             concentration.append(totaldataarray[k][0+2*i][j])
#         hold.plot(np.linspace(1,12,12), concentration, '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "First Runs/Fixed Time, Concentration/Singles/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'first runs at ' + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for second run, fixed time, concentration dependent
# print("Generating Graph Set 4...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         concentration = []
#         for k in range(12):
#             concentration.append(totaldataarray[k][1+2*i][j])
#         hold.plot(np.linspace(1,12,12), concentration, '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Second Runs/Fixed Time, Concentration/Singles/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'second runs at ' + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for overlapped, fixed concentration, time dependent
# print("Generating Graph Set 5...")
# start_time = time.time()
# color = cm.rainbow(np.linspace(0, 1, 11))

# for j in range(15):
#     tempfig = plt.figure()
#     hold = tempfig.add_subplot()
#     for i, c in enumerate(color):
        
#         areas = np.vstack([totaldataarray[i][0+2*j], totaldataarray[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             hold.plot(np.linspace(0,217,217), means, '-', c=c, label='C = 0 uM', linewidth=2)

#         else:
#             hold.plot(np.linspace(0,217,217), means, '-', markersize = 10, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM', linewidth=5.0)

#         hold.fill_between(np.linspace(0,217,217), means - stds, means + stds,color=c, alpha=0.4)
#         hold.set_ylim([0.7, 1.2])
#         hold.set_xticks(np.linspace(0, 217, 217),fontweight='bold')
#         hold.set_yticks(np.linspace(0.7, 1.2, 5),fontweight='bold')
#         hold.set_xticklabels([int(number) for number in (np.linspace(0,18,217))],fontweight='bold')
#         hold.set_yticklabels(np.around(np.arange(0.7, 1.2, 0.1),2),fontweight='bold')
#         hold.tick_params(axis='both', labelsize=20)
#         hold.locator_params(nbins = 3, axis='both')
#         hold.spines[['right', 'top']].set_visible(False)
#     newpath = newpath1 + "Overlapped Runs/Fixed Concentration, Time/" 
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     filename = newpath + 'overlapped' + 'concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for overlapped, fixed time, concentration dependent
# print("Generating Graph Set 6...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         concentration1 = []
#         concentration2 = []
#         for k in range(12):
#             concentration1.append(totaldataarray[k][0+2*i][j])
#             concentration2.append(totaldataarray[k][1+2*i][j])
#         hold.plot(np.linspace(1,12,12), concentration1, '.', markersize = 3,color = 'blue')
#         hold.plot(np.linspace(1,12,12), concentration2, '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Overlapped Runs/Fixed Time, Concentration/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")


# # This is for averaged, fixed concentration, time dependent
# print("Generating Graph Set 7...")
# start_time = time.time()
# for i in range(12):
#     for j in range(15):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         hold.plot(np.linspace(0,217,217), (np.array(totaldataarray[i][0+2*j]) + totaldataarray[i][1+2*j]) / 2, '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Averaged Run/Fixed Concentration, Time/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for averaged, fixed time, concentration dependent
# print("Generating Graph Set 8...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         concentration = []
#         for k in range(12):
#             concentration.append((np.array(totaldataarray[k][0+2*i][j]) + totaldataarray[k][1+2*i][j]) / 2)
#         hold.plot(np.linspace(1,12,12), concentration, '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Averaged Run/Fixed Time, Concentration/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")


# # This is for averaged with error bands, fixed concentration, time dependent
# print("Generating Graph Set 9...")
# start_time = time.time()
# for i in range(12):
#     tempfig = plt.figure(figsize=(16.0, 10.0))
#     for j in range(15):
#         hold = tempfig.add_subplot(3,5,j+1)
#         areas = np.vstack([totaldataarray[i][0+2*j], totaldataarray[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         hold.plot(np.linspace(0,217,217), means, '-', markersize = 3,color = 'blue')
#         hold.fill_between(np.linspace(0,217,217), means - stds, means + stds, alpha=0.3)
#         hold.set_ylim([0, 1.5])
#         hold.spines[['right', 'top']].set_visible(False)
#         hold.title.set_text(strainlist2[j])
#     newpath = newpath1 + "Averaged with errors/Fixed Concentration, Time/" 
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)    
#     filename = newpath + 'array concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
#     tempfig.tight_layout(pad=3)
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for averaged with error bands with concentrations overlapped
# print("Generating Graph Set 10...")
# start_time = time.time()
# tempfig = plt.figure(figsize=(16, 10.0),layout = 'constrained')
# tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)
# color = cm.rainbow(np.linspace(0, 1, 12))

# for j in range(15):
#     hold = tempfig.add_subplot(3,5,j+1)
#     hold.set_title(strainlist2[j], fontweight = 'bold',fontsize=20)
#     for i, c in enumerate(color):

#         areas = np.vstack([totaldataarray[i][0+2*j], totaldataarray[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             hold.plot(np.linspace(0,217,217), means, '-', c=c, label='C = 0 uM', linewidth=2)

#         else:
#             hold.plot(np.linspace(0,217,217), means, '-', markersize = 10, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM', linewidth=5.0)

#         hold.fill_between(np.linspace(0,217,217), means - stds, means + stds,color=c, alpha=0.4)
#         hold.set_ylim([0.3, 5.5])
#         hold.set_xticks(np.linspace(0, 217, 217),fontweight='bold')
#         hold.set_yticks(np.linspace(0.3, 5.5,10),fontweight='bold')
#         hold.set_xticklabels([int(number) for number in (np.linspace(0,18,217))],fontweight='bold')
#         hold.set_yticklabels(np.around(np.arange(0.3, 5.5,0.52),2),fontweight='bold')
#         hold.tick_params(axis='both', labelsize=20)
#         hold.locator_params(nbins = 3, axis='both')
#         hold.spines[['right', 'top']].set_visible(False)
# newpath = newpath1 + "Averaged with errors/Fixed Concentration, Time/" 
# if not os.path.exists(newpath):
#     os.makedirs(newpath)           
# filename = newpath + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is for averaged with error bands, fixed time, concentration dependent
# print("Generating Graph Set 11...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         cmean = []
#         clow = []
#         chigh = []
#         for k in range(12):
#             areas = np.vstack([totaldataarray[k][0+2*i][j] , totaldataarray[k][1+2*i][j]])
#             means = areas.mean(axis=0)
#             stds = areas.std(axis=0, ddof=1)
#             cmean.append(means[0])
#             clow.append(means[0]-stds[0])
#             chigh.append(means[0]+stds[0])
#         hold.plot(np.linspace(1,12,12), cmean, '.', markersize = 8,color = 'blue')
#         hold.fill_between(np.linspace(1,12,12), clow, chigh, alpha=0.7)
#         hold.set_ylim([0, 1.5])
#         newpath = newpath1 + "Averaged with errors/Fixed Concentration, Time/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)     
#         filename = 'C:/Users/danie/Desktop/NewDataset/Averaged with errors/Fixed Time, Concentration/' + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# This is array form of averaged with error bands, fixed time, concentration dependent
# print("Generating Graph Set 12...")
# start_time = time.time()
# for i in range(numberOfReads):
#     a = 0
#     b = 0
    
#     if i == 0 or i == 50 or i == 70 or i == 100 or i == 140 or i == 200:
#         fig, ax = plt.subplots(3,5,figsize=(24, 12),layout = 'constrained')
#         fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.2,hspace=0, wspace=0)
#         #fig.suptitle('Fold-Change Magnitudes At ' + str(i*5) + ' Minutes', fontsize=25)
#         #fig.supylabel('Fold Change Magnitude', fontsize=25)
#         #fig.supxlabel('Concentration (uM)', fontsize=25)
        
#         for j in range(15):
#             cmean = []
#             clow = []
#             chigh = []
#             for k in range(12):
#                 areas = np.vstack([totaldataarray[k][0+2*j][i] , totaldataarray[k][1+2*j][i]])
#                 means = areas.mean(axis=0)
#                 stds = areas.std(axis=0, ddof=1)
#                 cmean.append(means[0])
#                 clow.append(means[0]-stds[0])
#                 chigh.append(means[0]+stds[0])
#             ax[a,b].set_title(strainlist2[j], fontsize=30,pad=35, weight = 'bold')
#             ax[a,b].semilogx(np.logspace(11, 0, 12, base = 0.5), cmean, '.', markersize = 25,color = 'blue', base = 0.5)
#             ax[a,b].fill_between(np.logspace(11, 0, 12, base = 0.5), clow, chigh, alpha=0.7)
#             ax[a,b].set_ylim([0.3, 6.3])
#             ax[a,b].spines[['right', 'top']].set_visible(False)
#             ax[a,b].set_xscale('log', base = 0.5)
#             ax[a,b].xaxis.set_major_formatter(ScalarFormatter())
#             ax[a,b].minorticks_off()
#             ax[a,b].set_xticks(np.logspace(11, 0, 12, base = 0.5),fontweight='bold')
#             ax[a,b].set_yticks(np.linspace(0.3,6.3,6),fontweight='bold')
#             #ax[a,b].set_xticklabels([['10',r'0.5$\bf{^1}$',r'0.5$\bf{^2}$',r'0.5$\bf{^3}$',r'0.5$\bf{^4}$',r'0.5$\bf{^5}$',r'0.5$\bf{^6}$',r'0.5$\bf{^7}$',r'0.5$\bf{^8}$',r'0.5$\bf{^9}$',r'0.5$\bf{^10}$','0',]],fontweight='bold')
#             ax[a,b].set_xticklabels([str(round(10*(0.5**i),2)) for i in range(12)],fontweight='bold')
            
#             ax[a,b].set_yticklabels(np.arange(0,6,1),fontweight='bold')
#             # ax[a,b].tick_params(axis='both', labelsize=20)
#             ax[a,b].locator_params(nbins = 2, axis='both')
#             ax[a,b].tick_params(axis='both', labelsize=30)
            
#             b = b+1
            
#             if b == 5:
#                 b = 0
#                 a = a+1
#         newpath = newpath1 + "Averaged with errors/Fixed Concentration, Time/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)   
#         filename = newpath + 'time' + str(i) + ' grid.png'
#         fig.savefig(filename)
#         plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")



# #This is for overlapped OD600, same style as the previous overlapped graph
# print("Generating Graph Set 13...")
# start_time = time.time()
# fig, ax = plt.subplots(3,5,figsize=(24, 12))
# fig.subplots_adjust(wspace = 0.5,hspace = 1)
# color = cm.rainbow(np.linspace(0, 1, 12))
# lines = []
# a = 0
# b = 0
# for j in range(15):
    
    
#     ax[a,b].set_title(strainlist2[j], fontsize=35,pad=35)
    
#     for i, c in enumerate(color):

#         areas = np.vstack([od600dataarray[i][0+2*j], od600dataarray[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 10, c=c, label='C = 0 uM')
#         else:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 10, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM')
#         lines.append(ax[a,b].plot(np.linspace(0,217,217), means, '-', markersize = 6, c=c))
#         ax[a,b].fill_between(np.linspace(0,217,217), means - stds, means + stds, alpha=0.5)
#         ax[a,b].set_ylim([0, 1.5])
#         ax[a,b].locator_params(axis='y', nbins=2.5) 
#         ax[a,b].spines[['right', 'top']].set_visible(False)
#         ax[a,b].tick_params(axis='both', labelsize=35)
        
#     b = b+1
#     if b == 5:
#         b = 0
#         a = a+1
# newpath = newpath1 + "OD600/Overlapped/" 
# if not os.path.exists(newpath):
#     os.makedirs(newpath)   
# filename = newpath + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# ax[2,4].legend(bbox_to_anchor=(1.75, 3.75))

# fig.savefig(filename)
# plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")


# #This is for overlapped Raw1, same style as the previous overlapped graph
# print("Generating Graph Set 14...")
# start_time = time.time()
# fig, ax = plt.subplots(3,5,figsize=(24, 12))
# fig.subplots_adjust(wspace = 0.5,hspace = 1)
# color = cm.rainbow(np.linspace(0, 1, 12))
# lines = []
# a = 0
# b = 0
# for j in range(15):
    
    
#     ax[a,b].set_title(strainlist2[j], fontsize=35,pad=35)
    
#     for i, c in enumerate(color):

#         areas = np.vstack([rawnorm1array[i][0+2*j], rawnorm1array[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 1, c=c, label='C = 0 uM')
#         else:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 1, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM')
#         lines.append(ax[a,b].plot(np.linspace(0,217,217), means, '-', markersize = 1, c=c))
#         ax[a,b].fill_between(np.linspace(0,217,217), means - stds, means + stds, alpha=0.5, color = c)
#         # ax[a,b].set_ylim([0, 1.5])
#         ax[a,b].locator_params(axis='y', nbins=2.5) 
#         ax[a,b].spines[['right', 'top']].set_visible(False)
#         ax[a,b].tick_params(axis='both', labelsize=35)
        
#     b = b+1
#     if b == 5:
#         b = 0
#         a = a+1
# newpath = newpath1 + "Raw1/Overlapped/" 
# if not os.path.exists(newpath):
#     os.makedirs(newpath)   
# filename = newpath + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# ax[2,4].legend(bbox_to_anchor=(1.75, 3.75))

# fig.savefig(filename)
# plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# #This is for overlapped Raw2, same style as the previous overlapped graph
# print("Generating Graph Set 15...")
# start_time = time.time()
# fig, ax = plt.subplots(3,5,figsize=(24, 12))
# fig.subplots_adjust(wspace = 0.5,hspace = 1)
# color = cm.rainbow(np.linspace(0, 1, 12))
# lines = []
# a = 0
# b = 0
# for j in range(15):
    
    
#     ax[a,b].set_title(strainlist2[j], fontsize=35,pad=35)
    
#     for i, c in enumerate(color):

#         areas = np.vstack([rawnorm2array[i][0+2*j], rawnorm2array[i][1+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 1, c=c, label='C = 0 uM')
#         else:
#             ax[a,b].plot(np.linspace(0,217,217), means, '.', markersize = 1, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM')
#         lines.append(ax[a,b].plot(np.linspace(0,217,217), means, '-', markersize = 1, c=c))
#         ax[a,b].fill_between(np.linspace(0,217,217), means - stds, means + stds, alpha=0.5)
#         # ax[a,b].set_ylim([0, 1.5])
#         ax[a,b].locator_params(axis='y', nbins=2.5) 
#         ax[a,b].spines[['right', 'top']].set_visible(False)
#         ax[a,b].tick_params(axis='both', labelsize=35)
        
#     b = b+1
#     if b == 5:
#         b = 0
#         a = a+1
# newpath = newpath1 + "Raw2/Overlapped/" 
# if not os.path.exists(newpath):
#     os.makedirs(newpath)   
# filename = newpath1 + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# ax[2,4].legend(bbox_to_anchor=(1.75, 3.75))

# fig.savefig(filename)
# plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")



# # This is for OD600 with error bands, fixed time, concentration dependent
# print("Generating Graph Set 16...")
# start_time = time.time()
# for i in range(15):
#     for j in range(numberOfReads):
#         tempfig = plt.figure()
#         hold = tempfig.add_subplot()
#         cmean = []
#         clow = []
#         chigh = []
#         for k in range(12):
#             areas = np.vstack([od600dataarray[k][0+2*i][j] , od600dataarray[k][1+2*i][j]])
#             means = areas.mean(axis=0)
#             stds = areas.std(axis=0, ddof=1)
#             cmean.append(means[0])
#             clow.append(means[0]-stds[0])
#             chigh.append(means[0]+stds[0])
#         hold.plot(np.linspace(1,12,12), cmean, '.', markersize = 8,color = 'blue')
#         hold.fill_between(np.linspace(1,12,12), clow, chigh, alpha=0.7)
#         hold.set_ylim([0, 1.5])
#         newpath = newpath1 + "OD600/Fixed Time, Concentration/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)  
#         filename = newpath + 'time' + str(j) + ' strain ' + str(i) + 'concentration 0.5^' + str(k) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# #This is array form of OD600 with error bands, fixed time, concentration dependent
# print("Generating Graph Set 17...")
# start_time = time.time()

# for i in range(numberOfReads):
#     a = 0
#     b = 0
    
#     if i == 0 or i == 50 or i == 70 or i == 100 or i == 140 or i == 200:
#         fig, ax = plt.subplots(3,5,figsize=(24, 12),layout = 'constrained')
#         fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.2,hspace=0, wspace=0)
#         #fig.suptitle('Fold-Change Magnitudes At ' + str(i*5) + ' Minutes', fontsize=25)
#         #fig.supylabel('Fold Change Magnitude', fontsize=25)
#         #fig.supxlabel('Concentration (uM)', fontsize=25)
        
#         for j in range(15):
#             cmean = []
#             clow = []
#             chigh = []
#             for k in range(12):
#                 areas = np.vstack([od600dataarray[k][0+2*j][i] , od600dataarray[k][1+2*j][i]])
#                 means = areas.mean(axis=0)
#                 stds = areas.std(axis=0, ddof=1)
#                 cmean.append(means[0])
#                 clow.append(means[0]-stds[0])
#                 chigh.append(means[0]+stds[0])
#             ax[a,b].set_title(strainlist2[j], fontsize=30,pad=35, weight = 'bold')
#             ax[a,b].semilogx(np.logspace(11, 0, 12, base = 0.5), cmean, '.', markersize = 25,color = 'blue', base = 0.5)
#             ax[a,b].fill_between(np.logspace(11, 0, 12, base = 0.5), clow, chigh, alpha=0.7)
#             ax[a,b].set_ylim([0, 1.5])
#             ax[a,b].spines[['right', 'top']].set_visible(False)
#             ax[a,b].set_xscale('log', base = 0.5)
#             ax[a,b].xaxis.set_major_formatter(ScalarFormatter())
#             ax[a,b].minorticks_off()
#             ax[a,b].set_xticks(np.logspace(11, 0, 12, base = 0.5),fontweight='bold')
#             ax[a,b].set_yticks(np.linspace(0,1.5,10),fontweight='bold')
#             #ax[a,b].set_xticklabels([['10',r'0.5$\bf{^1}$',r'0.5$\bf{^2}$',r'0.5$\bf{^3}$',r'0.5$\bf{^4}$',r'0.5$\bf{^5}$',r'0.5$\bf{^6}$',r'0.5$\bf{^7}$',r'0.5$\bf{^8}$',r'0.5$\bf{^9}$',r'0.5$\bf{^10}$','0',]],fontweight='bold')
#             ax[a,b].set_xticklabels([str(round(10*(0.5**i),2)) for i in range(12)],fontweight='bold')
            
#             ax[a,b].set_yticklabels(np.arange(0,1.5,0.15),fontweight='bold')
#             # ax[a,b].tick_params(axis='both', labelsize=20)
#             ax[a,b].locator_params(nbins = 2, axis='both')
#             ax[a,b].tick_params(axis='both', labelsize=30)
            
#             b = b+1
            
#             if b == 5:
#                 b = 0
#                 a = a+1
#         newpath = newpath1 + "OD600/Fixed Time, Concentration/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)   
#         filename = newpath + 'time' + str(i) + ' grid.png'
#         fig.savefig(filename)
#         plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")

# # This is array form of Raw1 with error bands, fixed time, concentration dependent
# print("Generating Graph Set 18...")
# start_time = time.time()

# for i in range(numberOfReads):
#     a = 0
#     b = 0
    
#     if i == 0 or i == 50 or i == 70 or i == 100 or i == 140 or i == 200:
#         fig, ax = plt.subplots(3,5,figsize=(24, 12),layout = 'constrained')
#         fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.2,hspace=0, wspace=0)
#         #fig.suptitle('Fold-Change Magnitudes At ' + str(i*5) + ' Minutes', fontsize=25)
#         #fig.supylabel('Fold Change Magnitude', fontsize=25)
#         #fig.supxlabel('Concentration (uM)', fontsize=25)
        
#         for j in range(15):
#             cmean = []
#             clow = []
#             chigh = []
#             for k in range(12):
#                 areas = np.vstack([rawnorm1array[k][0+2*j][i] , rawnorm1array[k][1+2*j][i]])
#                 means = areas.mean(axis=0)
#                 stds = areas.std(axis=0, ddof=1)
#                 cmean.append(means[0])
#                 clow.append(means[0]-stds[0])
#                 chigh.append(means[0]+stds[0])
#             ax[a,b].set_title(strainlist2[j], fontsize=30,pad=35, weight = 'bold')
#             ax[a,b].semilogx(np.logspace(11, 0, 12, base = 0.5), cmean, '.', markersize = 25,color = 'blue', base = 0.5)
#             ax[a,b].fill_between(np.logspace(11, 0, 12, base = 0.5), clow, chigh, alpha=0.7)
#             # ax[a,b].set_ylim([0, 1.5])
#             ax[a,b].spines[['right', 'top']].set_visible(False)
#             ax[a,b].set_xscale('log', base = 0.5)
#             ax[a,b].xaxis.set_major_formatter(ScalarFormatter())
#             ax[a,b].minorticks_off()
#             ax[a,b].set_xticks(np.logspace(11, 0, 12, base = 0.5),fontweight='bold')
#             ax[a,b].set_yticks(np.linspace(0,1.5,10),fontweight='bold')
#             #ax[a,b].set_xticklabels([['10',r'0.5$\bf{^1}$',r'0.5$\bf{^2}$',r'0.5$\bf{^3}$',r'0.5$\bf{^4}$',r'0.5$\bf{^5}$',r'0.5$\bf{^6}$',r'0.5$\bf{^7}$',r'0.5$\bf{^8}$',r'0.5$\bf{^9}$',r'0.5$\bf{^10}$','0',]],fontweight='bold')
#             ax[a,b].set_xticklabels([str(round(10*(0.5**i),2)) for i in range(12)],fontweight='bold')
            
#             ax[a,b].set_yticklabels(np.arange(0,1.5,0.15),fontweight='bold')
#             # ax[a,b].tick_params(axis='both', labelsize=20)
#             ax[a,b].locator_params(nbins = 2, axis='both')
#             ax[a,b].tick_params(axis='both', labelsize=30)
            
#             b = b+1
            
#             if b == 5:
#                 b = 0
#                 a = a+1
#         newpath = newpath1 + "Raw1/Fixed Time, Concentration/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)   
#         filename = newpath + 'time' + str(i) + ' grid.png'
#         fig.savefig(filename)
#         plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")


# print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - startTOTAL_time))








#######################################################



# # SPECIAL CASE

# newpath = r'C:\Users\danie\Desktop\Experiment1' 

# strainlist22 = ["sucC", "cspA2", "gitA","lpxC"]

# tempfig = plt.figure(figsize=(16, 10.0),layout = 'constrained')
# tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)
# color = cm.rainbow(np.linspace(0, 1, 12))

# for j in range(4):
#     hold = tempfig.add_subplot(2,2,j+1)
#     hold.set_title(strainlist22[j], fontweight = 'bold',fontsize=20)
#     for i, c in enumerate(color):

#         areas = np.vstack([totaldataarray[i][0+2*j], totaldataarray[i][1+2*j],totaldataarray[i][2+2*j], totaldataarray[i][3+2*j]])
#         means = areas.mean(axis=0)
#         stds = areas.std(axis=0, ddof=1)
#         if i == 11:
#             hold.plot(np.linspace(0,217,217), means, '-', c=c, label='C = 0 uM', linewidth=2)

#         else:
#             hold.plot(np.linspace(0,217,217), means, '-', markersize = 1, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM', linewidth=1.0)

#         hold.fill_between(np.linspace(0,217,217), means - stds, means + stds,color=c, alpha=0.4)
#         hold.set_ylim([0.6, 1.2])
#         hold.set_xticks(np.linspace(0, 217, 217),fontweight='bold')
#         hold.set_yticks(np.linspace(0.6,1.2,10),fontweight='bold')
#         hold.set_xticklabels([int(number) for number in (np.linspace(0,18,217))],fontweight='bold')
#         hold.set_yticklabels(np.around(np.arange(0.6,1.2,0.06),2),fontweight='bold')
#         hold.tick_params(axis='both', labelsize=20)
#         hold.locator_params(nbins = 3, axis='both')
#         hold.spines[['right', 'top']].set_visible(False)
        
# filename = 'C:/Users/danie/Desktop/NewDataSet/Fidelities/Fixed Concentration, Time/' + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)




# # # This is array form of averaged with error bands, fixed time, concentration dependent


# strainlist22 = ["sucC", "cspA2", "gitA","lpxC"]

# for i in range(numberOfReads):
#     a = 0
#     b = 0
    
#     if i == 0 or i == 50 or i == 70 or i == 100 or i == 140 or i == 200:
#         fig, ax = plt.subplots(2,2,figsize=(24, 12),layout = 'constrained')
#         fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.2,hspace=0, wspace=0)
#         #fig.suptitle('Fold-Change Magnitudes At ' + str(i*5) + ' Minutes', fontsize=25)
#         #fig.supylabel('Fold Change Magnitude', fontsize=25)
#         #fig.supxlabel('Concentration (uM)', fontsize=25)
        
#         for j in range(4):
#             cmean = []
#             clow = []
#             chigh = []
#             for k in range(12):
#                 areas = np.vstack([totaldataarray[k][0+2*j][i] , totaldataarray[k][1+2*j][i], totaldataarray[k][2+2*j][i] , totaldataarray[k][3+2*j][i]])
#                 means = areas.mean(axis=0)
#                 stds = areas.std(axis=0, ddof=1)
#                 cmean.append(means[0])
#                 clow.append(means[0]-stds[0])
#                 chigh.append(means[0]+stds[0])
#             ax[a,b].set_title(strainlist22[j], fontsize=30,pad=35, weight = 'bold')
#             ax[a,b].semilogx(np.logspace(11, 0, 12, base = 0.5), cmean, '.', markersize = 25,color = 'blue', base = 0.5)
#             ax[a,b].fill_between(np.logspace(11, 0, 12, base = 0.5), clow, chigh, alpha=0.7)
#             ax[a,b].set_ylim([0, 1.5])
#             ax[a,b].spines[['right', 'top']].set_visible(False)
#             ax[a,b].set_xscale('log', base = 0.5)
#             ax[a,b].xaxis.set_major_formatter(ScalarFormatter())
#             ax[a,b].minorticks_off()
#             ax[a,b].set_xticks(np.logspace(11, 0, 12, base = 0.5),fontweight='bold')
#             ax[a,b].set_yticks(np.linspace(0.6,1.0,10),fontweight='bold')
#             #ax[a,b].set_xticklabels([['10',r'0.5$\bf{^1}$',r'0.5$\bf{^2}$',r'0.5$\bf{^3}$',r'0.5$\bf{^4}$',r'0.5$\bf{^5}$',r'0.5$\bf{^6}$',r'0.5$\bf{^7}$',r'0.5$\bf{^8}$',r'0.5$\bf{^9}$',r'0.5$\bf{^10}$','0',]],fontweight='bold')
#             ax[a,b].set_xticklabels([str(round(10*(0.5**i),2)) for i in range(12)],fontweight='bold')
            
#             ax[a,b].set_yticklabels(np.arange(0.6,1.0,0.04),fontweight='bold')
#             # ax[a,b].tick_params(axis='both', labelsize=20)
#             ax[a,b].locator_params(nbins = 2, axis='both')
#             ax[a,b].tick_params(axis='both', labelsize=30)
            
#             b = b+1
            
#             if b == 2:
#                 b = 0
#                 a = a+1
#         filename = 'C:/Users/danie/Desktop/NewDataset/Fidelities/Fixed Time, Concentration/' + 'time' + str(i) + ' grid.png'
#         fig.savefig(filename)
#         plt.close(fig)

# # This is for first run, fixed concentration, time dependent
# i = 0
# j = 3
# tempfig = plt.figure()
# hold = tempfig.add_subplot()
# hold.plot(np.linspace(0,217,217), rawcntrlf[i][0+2*j], '-', markersize = 3,color = 'blue')
# hold.plot(np.linspace(0,217,217), rawcntrlf[i][1+2*j], '-', markersize = 3,color = 'red')
# hold.plot(np.linspace(0,217,217), rawcntrlf[i][2+2*j], '-', markersize = 3,color = 'green')
# hold.plot(np.linspace(0,217,217), rawcntrlf[i][3+2*j], '-', markersize = 3,color = 'orange')
# filename = 'C:/Users/danie/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolfconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)

# i = 0
# j = 3
# tempfig = plt.figure()
# hold = tempfig.add_subplot()
# hold.plot(np.linspace(0,217,217), rawcntrlod[i][0+2*j], '-', markersize = 3,color = 'blue')
# hold.plot(np.linspace(0,217,217), rawcntrlod[i][1+2*j], '-', markersize = 3,color = 'red')
# hold.plot(np.linspace(0,217,217), rawcntrlod[i][2+2*j], '-', markersize = 3,color = 'green')
# hold.plot(np.linspace(0,217,217), rawcntrlod[i][3+2*j], '-', markersize = 3,color = 'orange')
# filename = 'C:/Users/danie/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolodconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)

# i = 0
# j = 3
# tempfig = plt.figure()
# hold = tempfig.add_subplot()
# hold.plot(np.linspace(0,217,217), rawcntrl[i][0+2*j], '-', markersize = 3,color = 'blue')
# hold.plot(np.linspace(0,217,217), rawcntrl[i][1+2*j], '-', markersize = 3,color = 'red')
# hold.plot(np.linspace(0,217,217), rawcntrl[i][2+2*j], '-', markersize = 3,color = 'green')
# hold.plot(np.linspace(0,217,217), rawcntrl[i][3+2*j], '-', markersize = 3,color = 'orange')
# filename = 'C:/Users/danie/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)

# i = 0
# j = 3
# tempfig = plt.figure()
# hold = tempfig.add_subplot()
# hold.plot(np.linspace(0,217,217), totaldataarray[i][0+2*j], '-', markersize = 3,color = 'blue')
# hold.plot(np.linspace(0,217,217), totaldataarray[i][1+2*j], '-', markersize = 3,color = 'red')
# hold.plot(np.linspace(0,217,217), totaldataarray[i][2+2*j], '-', markersize = 3,color = 'green')
# hold.plot(np.linspace(0,217,217), totaldataarray[i][3+2*j], '-', markersize = 3,color = 'orange')
# filename = 'C:/Users/danie/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusonfoldchangeconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
# tempfig.savefig(filename)
# plt.close(tempfig)



                