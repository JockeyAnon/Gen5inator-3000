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
abrod = np.array([])
abrf = np.array([])
stdbrod = np.array([])
stdbrf = np.array([])
stdbsod = np.array([])
stdbsf = np.array([])
absod = np.array([])
absf = np.array([])
phonebookpage1 = np.array([])
phonebookpage2 = np.array([])
phonebookpage3 = np.array([])
phonebookpage4 = np.array([])
phonebookpage5 = np.array([])
phonebookpage6 = np.array([])
phonebookpage7 = np.array([])

graphcounter = 1


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
od600dataarray = np.delete(od600dataarraywc, np.s_[11::12], 0)
fluordataarray = np.delete(fluordataarraywc, np.s_[11::12], 0)
rawnorm = np.divide(fluordataarray,od600dataarray)
rawnormwc = np.divide(fluordataarraywc,od600dataarraywc)
rawnormc = np.divide(controlfluor,controlod600)

od600dataarray = np.array_split(od600dataarray, 8)
od600dataarraywc = np.array_split(od600dataarraywc, 8)
fluordataarray = np.array_split(fluordataarray, 8)
fluordataarraywc = np.array_split(fluordataarraywc, 8)





c=0
for i in range(8):
    for j in range(12):
        fc1 = np.append(fc1,np.divide(rawnormwc[c],rawnormc[i]))
        c+=1

rawnorm = np.array_split(rawnorm, 8)
rawnormwc = np.array_split(rawnormwc, 8)
rawnormc = np.array_split(rawnormc, 8)

fc1 = np.array_split(np.array_split(fc1,96),8)
fc2 = [np.delete(i, 11, 0) for i in fc1]



averagebyrowfc1 = [np.mean(i,axis=0) for i in np.array_split(fc1, 4)]
abrstdfc1 = [np.std(i,axis=0) for i in np.array_split(fc1, 4)]
averagefc1pair = np.array([averagebyrowfc1,abrstdfc1])

averagebyrowfc2 = [np.mean(i,axis=0) for i in np.array_split(fc2, 4)]
abrstdfc2 = [np.std(i,axis=0) for i in np.array_split(fc2, 4)]
averagefc2pair = np.array([averagebyrowfc2,abrstdfc2])

averagebysquares = [[np.mean(j,axis=0) for j in np.array_split(i, 6)] for i in averagebyrowfc1]
absstd = [[np.std(j,axis=0) for j in np.array_split(i, 6)] for i in averagebyrowfc1]
averagebysquarepair = np.array([averagebysquares,absstd])

abrod = [np.mean(i,axis=0) for i in np.array_split(od600dataarraywc, 4)]
stdbrod = [np.std(i,axis=0) for i in np.array_split(od600dataarraywc, 4)]
averageod = np.array([abrod,stdbrod])

abrf = [np.mean(i,axis=0) for i in np.array_split(fluordataarraywc, 4)]
stdbrf = [np.std(i,axis=0) for i in np.array_split(fluordataarraywc, 4)]
averagef = np.array([abrf,stdbrf])

absod = [np.mean(i,axis=0) for i in np.array_split(abrod, 4)]
stdbsod = [np.std(i,axis=0) for i in np.array_split(abrod, 4)]
averagebsod = np.array([absod,stdbsod])

absf = [np.mean(i,axis=0) for i in np.array_split(abrf, 4)]
stdbsf = [np.std(i,axis=0) for i in np.array_split(abrf, 4)]
averagebsf = np.array([absf,stdbsf])

phonebookpage1 = np.vstack([od600dataarraywc,fluordataarraywc,rawnormwc,fc1])
phonebookpage4 = np.vstack([od600dataarray,fluordataarray,rawnorm,fc2])
phonebookpage2 = np.vstack([controlod600,controlfluor])
phonebookpage5 = np.vstack([rawnormc])
phonebookpage6 = np.vstack([averagefc2pair])
phonebookpage7 = np.vstack([averagebysquarepair])
phonebookpage3 = np.vstack([averagefc1pair,averageod,averagef,averagebsod,averagebsf])


print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - start_time))


newpath1 = newdir + "Graphics/"     
if not os.path.exists(newpath1):
    os.makedirs(newpath1)
length = len(fc1[0][0])

def var_name(var):
    for name, value in globals().items():
        if value is var:
            return name



def graph(source,type,averaged,errors,lod,smoothing,rte,graphcounter):
    print("Generating Graph Set " + str(graphcounter) + "...")
    start_time = time.time()
    newpath = newpath1 + str(var_name(source)) + "/Fixed Concentration, Time/"
    unittype = ''
    if lod == 0:
        unittype = '.'
        newpath+="dots/"
    else:
        unittype = '-'
        newpath+="lines/"
    plt.rcParams['font.weight'] = 'bold'
    if averaged == 0:
        newpath+="unaveraged/"
        if errors == 0:
            newpath+="no errors/"
            dim1=len(source)
            dim2=len(source[0])
            color = cm.rainbow(np.linspace(0, 1, dim2))
            ssource = signal.savgol_filter(source,50,1)
            if type == 0:
                fig, ax = plt.subplots(nrows=dim1, ncols=dim2,figsize=(32, 20))
                fig.tight_layout(pad=5)
                newpath+="array/"
            for i in range(dim1):
                if type == 1:
                    tempfig = plt.figure(figsize=(32, 20))
                    newpath+="stacked/"
                for j , c in enumerate(color):
                    if type == 2:
                        tempfig = plt.figure(figsize=(32, 20))
                        newpath+="singles/"
                    if smoothing == 0:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), source[i], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), source[i], unittype, markersize = 3, color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source]),max(x for x in [y for y in source])])
                        newpath+="not smoothed/"
                    if smoothing == 1:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), ssource[i], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), ssource[i], unittype, markersize = 3, color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source]),max(x for x in [y for y in source])])
                        newpath+="smoothed/"
                    if type == 2:
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                        tempfig.savefig(filename)
                        plt.close(tempfig)
                if type == 1:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                    tempfig.savefig(filename)
                    plt.close(tempfig)
            if type == 0:
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                fig.savefig(filename)
                plt.close(fig)
        if errors == 1:
            print("This data set is not averaged, thus it has no error set!")
        else:
            print("Invalid graph gen request codes")
    elif averaged == 1:
        newpath+="averaged/"
        if errors == 0:
            newpath+="no errors/"
            dim1=len(source[0])
            dim2=len(source[0][0])
            color = cm.rainbow(np.linspace(0, 1, dim2))
            ssource = signal.savgol_filter(source[0],50,1)
            if type == 0:
                fig, ax = plt.subplots(nrows=dim1, ncols=dim2,figsize=(32, 20))
                fig.tight_layout(pad=5)
                newpath+="array/"
            for i in range(dim1):
                if type == 1:
                    tempfig = plt.figure(figsize=(32, 20))
                    newpath+="stacked/"
                for j , c in enumerate(color):
                    if type == 2:
                        tempfig = plt.figure(figsize=(32, 20))
                        newpath+="singles/"
                    if smoothing == 0:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source]),max(x for x in [y for y in source])])
                        newpath+="not smoothed/"
                    if smoothing == 1:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), ssource[i][j], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), ssource[i][j], unittype, markersize = 3, color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source[0]]),max(x for x in [y for y in source[0]])])
                        newpath+="smoothed/"
                    if type == 2:
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                        tempfig.savefig(filename)
                        plt.close(tempfig)
                if type == 1:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                    tempfig.savefig(filename)
                    plt.close(tempfig)
            if type == 0:
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                fig.savefig(filename)
                plt.close(fig)
        elif errors == 1:
            highs = np.add(source[0],source[1])
            lows = np.subtract(source[0],source[1])
            shighs = signal.savgol_filter(np.add(source[0],source[1]),50,1)
            slows = signal.savgol_filter(np.subtract(source[0],source[1]),50,1)
            ssource = signal.savgol_filter(source[0],50,1)
            newpath+="no errors/"
            dim1=len(source[0])
            dim2=len(source[0][0])
            color = cm.rainbow(np.linspace(0, 1, dim2))
            if type == 0:
                fig, ax = plt.subplots(nrows=dim1, ncols=dim2,figsize=(32, 20))
                fig.tight_layout(pad=5)
                newpath+="array/"
            for i in range(dim1):
                if type == 1:
                    tempfig = plt.figure(figsize=(32, 20))
                    newpath+="stacked/"
                for j , c in enumerate(color):
                    if type == 2:
                        tempfig = plt.figure(figsize=(32, 20))
                        newpath+="singles/"
                    if smoothing == 0:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                            ax[i, j].fill_between(np.linspace(0,5,length), lows,highs, unittype, markersize = 3,color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source]),max(x for x in [y for y in source])])
                        newpath+="not smoothed/"
                    if smoothing == 1:
                        if type == 1 or type == 2:
                            tempfig.plot(np.linspace(0,rte,length), ssource[0][i][j], unittype, markersize = 3, color = c)
                        elif type == 0:
                            ax[i,j].plot(np.linspace(0,rte,length), ssource[0][i][j], unittype, markersize = 3, color = c)
                            ax[i, j].fill_between(np.linspace(0,5,length), slows,shighs, unittype, markersize = 3,color = c)
                            ax[i,j].set_ylim([min(x for x in [y for y in source]),max(x for x in [y for y in source])])
                        newpath+="smoothed/"
                    if type == 2:
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                        tempfig.savefig(filename)
                        plt.close(tempfig)
                if type == 1:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                    tempfig.savefig(filename)
                    plt.close(tempfig)
            if type == 0:
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                filename = newpath + 'graph row' + str(i) + ' column' + str(j) + '.png'
                fig.savefig(filename)
                plt.close(fig)
        else:
            print("Invalid graph gen request codes")
    print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
    print("Graphs Generated and Stored!")
    graphcounter+=1



for x in phonebookpage1:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage2:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage3:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage4:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage5:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage6:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1
for x in phonebookpage7:
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        graph(x,i,j,k,l,m,5,graphcounter)
                        graphcounter+=1







# # This is for first run, fixed concentration, time dependent
# print("Generating Graph Set " + str(graphcounter) + "...")
# start_time = time.time()
# for i in range(8):
#     for j in range(12):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), fc1[i][j], '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Singles/No Errors/WITH CONTROL/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# start_time = time.time()
# for i in range(8):
#     for j in range(11):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), fc2[i][j], '.', markersize = 3,color = 'blue')
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Singles/No Errors/NO CONTROL/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     tempfig = plt.figure()
#     for j , c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), fc1[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Stacked/No Errors/WITH CONTROL/dots/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# start_time = time.time()
# color = cm.rainbow(np.linspace(0, 1, 11))
# for i in range(8):
#     tempfig = plt.figure()
#     for j, c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), fc2[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Stacked/No Errors/NO CONTROL/dots/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     tempfig = plt.figure()
#     for j , c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), signal.savgol_filter(fc1[i][j], 50, 1), '-', markersize = 3,color = c)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Stacked/No Errors/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# start_time = time.time()
# color = cm.rainbow(np.linspace(0, 1, 11))
# for i in range(8):
#     tempfig = plt.figure()
#     for j, c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), signal.savgol_filter(fc1[i][j], 50, 1), '-', markersize = 3,color = c)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/Stacked/No Errors/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(4):
#     tempfig = plt.figure()
#     for j , c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), signal.savgol_filter(averagebyrowfc1[i][j], 50, 1), '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(averagebyrowfc1[i][j],abrstdfc1[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(averagebyrowfc1[i][j],abrstdfc1[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/StackedAvg/No Errors/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# start_time = time.time()
# color = cm.rainbow(np.linspace(0, 1, 11))
# for i in range(4):
#     tempfig = plt.figure()
#     for j, c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), signal.savgol_filter(averagebyrowfc2[i][j], 50, 1), '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(averagebyrowfc2[i][j],abrstdfc2[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(averagebyrowfc2[i][j],abrstdfc2[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/StackedAvg/With Errors/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 6))
# start_time = time.time()
# for i in range(4):
#     tempfig = plt.figure()
#     for j , c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), signal.savgol_filter(averagebysquares[i][j], 50, 1), '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(averagebysquares[i][j],absstd[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(averagebysquares[i][j],absstd[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,length,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "Fold Change/Fixed Concentration, Time/StackedAvgSqrs/With Errors/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'lines row' + str(i+1) + ' concentration 0.5^ ' + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarray[i][j], '-', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/NO CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1


# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarraywc[i][j], '-', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/WITH CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1


# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarray[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/NO CONTROL/dots/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/NO CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/NO CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1
# #START

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarraywc[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/WITH CONTROL/dots/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/WITH CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/WITH CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1



# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarray[i][j], '-', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/NO CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/NO CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1


# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarraywc[i][j], '-', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/WITH CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '-', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/WITH CONTROL/lines/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1


# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarray[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/NO CONTROL/dots/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/NO CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/NO CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1


# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), od600dataarraywc[i][j], '.', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/No Errors/WITH CONTROL/dots/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), abrod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(abrod[i][j],stdbrod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(abrod[i][j],stdbrod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageByRows/WITH CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 12))
# start_time = time.time()
# for i in range(8):
#     for j , c in enumerate(color):
#         tempfig = plt.figure()
#         plt.plot(np.linspace(0,5,length), absod[i][j], '.', markersize = 3,color = c)
#         highs = signal.savgol_filter(np.add(absod[i][j],stdbsod[i][j]), 50, 1)
#         lows = signal.savgol_filter(np.subtract(absod[i][j],stdbsod[i][j]), 50, 1)
#         plt.fill_between(np.linspace(0,5,length), highs, lows, color = c, alpha = 0.5)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Singles/AverageBySquares/WITH CONTROL/dots/smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#         tempfig.savefig(filename)
#         plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1



# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 11))
# start_time = time.time()
# for i in range(8):
#     tempfig = plt.figure()
#     for j , c in enumerate(color):
#         plt.plot(np.linspace(0,5,length), od600dataarray[i][j], '-', markersize = 3,color = c)
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Stacked/No Errors/NO CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#     filename = newpath + 'cell ' + str(chr(65+i)) + str(j+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 10))
# start_time = time.time()
# fig, ax = plt.subplots(nrows=8, ncols=10,figsize=(32, 20))
# fig.tight_layout(pad=5)
# plt.rcParams['font.weight'] = 'bold'
# for i in range(8):
#     for j , c in enumerate(color):
#         ax[i, j].plot(np.linspace(0,5,length), od600dataarray[i][j], '-', markersize = 3,color = 'blue')
#         ax[i, j].locator_params(axis='both', nbins=3)
#         ax[i, j].tick_params(axis='both', labelsize = 20)
#         for tick in ax[i, j].get_xticklabels():
#             tick.set_fontweight('bold')
#         for tick in ax[i, j].get_yticklabels():
#             tick.set_fontweight('bold')
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Plate/No Errors/NO CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
# filename = newpath + 'UNFIXED AXES cell ' + str(chr(65+i)) + str(j+1) + '.png'
# fig.savefig(filename)
# plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1

# print("Generating Graph Set " + str(graphcounter) + "...")
# color = cm.rainbow(np.linspace(0, 1, 10))
# start_time = time.time()
# fig, ax = plt.subplots(nrows=8, ncols=10,figsize=(32, 20))
# fig.tight_layout(pad=5)
# plt.rcParams['font.weight'] = 'bold'
# for i in range(8):
#     for j , c in enumerate(color):
#         ax[i, j].plot(np.linspace(0,5,length), od600dataarray[i][j], '-', markersize = 3,color = 'blue')
#         ax[i, j].set_ylim([0.1,0.25])
#         ax[i, j].locator_params(axis='both', nbins=3)
#         ax[i, j].tick_params(axis='both', labelsize = 20)
#         for tick in ax[i, j].get_xticklabels():
#             tick.set_fontweight('bold')
#         for tick in ax[i, j].get_yticklabels():
#             tick.set_fontweight('bold')
#         newpath = newpath1 + "OD600/Fixed Concentration, Time/Plate/No Errors/NO CONTROL/lines/not smoothed/" 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
# filename = newpath + 'FIXED AXES cell ' + str(chr(65+i)) + str(j+1) + '.png'
# fig.savefig(filename)
# plt.close(fig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")
# graphcounter+=1



                