#This is a bare-bones version of the calculation schema. There have been a lot of issues with data pointers in the past, with nested arrays starting the problem in the first place.
#For this attempt, we will keep the graph generation protocol mostly the same, but the calculation schema will be more well defined and consistent.
#Changing of package wrappers or data types will only be done when necessary, and will be prohibited mid calculation.

#Import Declarations
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
import matplotlib as mpl
import glob
from matplotlib.ticker import ScalarFormatter
import os
from matplotlib.pyplot import cm
from ast import literal_eval
import ast
import time
from matplotlib import pyplot
import statistics as st
import seaborn as sns
mpl.use('TkAgg')

rte = 7
length = 174

dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
val = "ExperimentPW"
newdir = "/Users/danielpark/Desktop/"+val+"/"
os.chdir(newdir)
path = os.getcwd() 
f = os.listdir(path)


print("No Previous Calculated Data Detected!")
start_time = time.time()
csv_files = glob.glob(os.path.join(path, "*.xlsx")) 

#Turns all the xlsx spreadsheets into dataframes and transposed to make index calls easier
for f in csv_files:
    dataframes.append(pd.read_excel(f).values.transpose())

od600 = dataframes[0]
fluor = dataframes[1]
# fluor2 = dataframes[2]
rawnorm = [i / j for i, j in zip(fluor, od600)]

#To determine suitable replicates to use for displaying the data for the pulse width input data,
#   we will first split the dataset into groups of 4 wells, since the experiment was replicated across
#   pairs of rows and columns
n = 2
wellpairs = [rawnorm[i:i + n] for i in range(0, len(rawnorm), n)]
quadrantlist = []
for i in range(42):
    quadrantlist.append([wellpairs[i],wellpairs[i+6]])
n=6
quadrantgroups = [quadrantlist[i:i + n] for i in range(0, len(quadrantlist), n)]
del quadrantgroups[1::2]
quadrantregroups = [[np.array([j[0][0],j[0][1],j[1][0],j[1][1]])for j in i] for i in quadrantgroups]
quadrantgroupscov = [[np.cov(np.array([j[0][0],j[0][1],j[1][0],j[1][1]]),bias=False)for j in i] for i in quadrantgroups]

# fig, ax = plt.subplots(nrows=8, ncols=12,figsize=(32, 20))
# fig.tight_layout(pad=5)
# k=0
# for i in range(8):
#     for j in range(12):
#         ax[i,j].plot(np.linspace(0,rte,length), rawnorm[k], '-', markersize = 3)
#         k+=1
# plt.show()

# labs = ['left top', 'right top', 'left bottom', 'right bottom']
# sns.heatmap(quadrantgroupscov[1][0], annot=True, fmt='g', xticklabels=labs, yticklabels=labs)
# plt.show()




# listofsimilarones = []
# for i in quadrantgroupscov:
#     for j in i:
#         lt = j[-1][0] * 10**11
#         rt = j[-1][1] * 10**11
#         lb = j[-1][2] * 10**11
#         rb = j[-1][3] * 10**11

#         differencelist = np.argsort(np.array([abs(lt-rt),abs(lt-lb),abs(lt-rb),abs(rt-lb),abs(rt-rb),abs(lb-rb)]))
#         for k in range(6):
#             if differencelist[0]==0:
#                 listofsimilarones.append([1,1,0,0])
#                 break
#             elif differencelist[0]==1:
#                 listofsimilarones.append([1,0,1,0])
#                 break
#             elif differencelist[0]==2:
#                 listofsimilarones.append([1,0,0,1])
#                 break
#             elif differencelist[0]==3:
#                 listofsimilarones.append([0,1,1,0])
#                 break
#             elif differencelist[0]==4:
#                 listofsimilarones.append([0,1,0,1])
#                 break
#             elif differencelist[0]==5:
#                 listofsimilarones.append([0,0,1,1])
#                 break

# listofsimilarones = []
# for i in quadrantregroups:
#     for j in i:
#         differencelist = np.argsort(np.array([np.std(np.std(np.array([j[0],j[1],j[2]]),axis=0),axis=0),np.std(np.std(np.array([j[0],j[1],j[3]]),axis=0),axis=0),np.std(np.std(np.array([j[0],j[2],j[3]]),axis=0),axis=0),np.std(np.std(np.array([j[1],j[2],j[3]]),axis=0),axis=0)]))
#         for k in range(4):
#             if differencelist[0]==0:
#                 listofsimilarones.append([1,1,1,0])
#                 break
#             elif differencelist[0]==1:
#                 listofsimilarones.append([1,1,0,1])
#                 break
#             elif differencelist[0]==2:
#                 listofsimilarones.append([1,0,1,1])
#                 break
#             elif differencelist[0]==3:
#                 listofsimilarones.append([0,1,1,1])
#                 break

mostsimilarthrees = []
for i in quadrantregroups:
    for j in i:
        differencelist = np.argsort(np.array([np.std(np.std(np.array([j[0],j[1],j[2]]),axis=0),axis=0),np.std(np.std(np.array([j[0],j[1],j[3]]),axis=0),axis=0),np.std(np.std(np.array([j[0],j[2],j[3]]),axis=0),axis=0),np.std(np.std(np.array([j[1],j[2],j[3]]),axis=0),axis=0)]))
        for k in range(4):
            if differencelist[0]==0:
                mostsimilarthrees.append([j[0],j[1],j[2]])
                break
            elif differencelist[0]==1:
                mostsimilarthrees.append([j[0],j[1],j[3]])
                break
            elif differencelist[0]==2:
                mostsimilarthrees.append([j[0],j[1],j[3]])
                break
            elif differencelist[0]==3:
                mostsimilarthrees.append([j[1],j[2],j[3]])
                break



# listofsimilarones = []
# for i in quadrantregroups:
#     for j in i:
#         lt = j[0]
#         rt = j[1]
#         lb = j[2]
#         rb = j[3]

        
#         differencelist = np.argsort(np.array([np.mean(np.std(np.vstack((lt,rt)),axis=0)),np.mean(np.std(np.vstack((lt,lb)),axis=0)), np.mean(np.std(np.vstack((lt,rb)),axis=0)) ,np.mean(np.std(np.vstack((rt,lb)),axis=0)), np.mean(np.std(np.vstack((rt,rb)),axis=0)), np.mean(np.std(np.vstack((lb,rb)),axis=0))]))
#         for k in range(6):
#             if differencelist[0]==0:
#                 listofsimilarones.append([1,1,0,0])
#                 break
#             elif differencelist[0]==1:
#                 listofsimilarones.append([1,0,1,0])
#                 break
#             elif differencelist[0]==2:
#                 listofsimilarones.append([1,0,0,1])
#                 break
#             elif differencelist[0]==3:
#                 listofsimilarones.append([0,1,1,0])
#                 break
#             elif differencelist[0]==4:
#                 listofsimilarones.append([0,1,0,1])
#                 break
#             elif differencelist[0]==5:
#                 listofsimilarones.append([0,0,1,1])
#                 break

# orderedlist = np.array([])
# counter = 0
# first = np.array([])
# second = np.array([])
# for i in listofsimilarones:
#     first = np.append(first,i[0])
#     first = np.append(first,i[1])
#     second = np.append(second,i[2])
#     second = np.append(second,i[3])
#     if len(first)==12:
#         orderedlist = np.append(orderedlist,first)
#         orderedlist = np.append(orderedlist,second)
#         first = np.array([])
#         second = np.array([])


# fig, ax = plt.subplots(nrows=4, ncols=6,figsize=(32, 20))
# fig.tight_layout(pad=5)
# k=0
# for i in range(4):
#     for j in range(6):
#         ax[i,j].plot(np.linspace(0,rte,length), quadrantregroups[k], '-', markersize = 3)
#         k+=1
# plt.show()


# labs = ['left top', 'right top', 'left bottom', 'right bottom']
# sns.heatmap(quadrantgroupscov[1][0], annot=True, fmt='g', xticklabels=labs, yticklabels=labs)
# plt.show()

# print("Done")


    

# fig, ax = plt.subplots(nrows=8, ncols=12,figsize=(32, 20))
# fig.tight_layout(pad=5)
# k=0
# for i in range(8):
#     for j in range(12):
#         ax[i,j].plot(np.linspace(0,rte,length), rawnorm[k], '-', markersize = 3)
#         if orderedlist[k] == 1:
#             ax[i,j].set_facecolor('blue')
#         k+=1
        
# plt.show()

# onlyrelevants = np.array([])
# for i in range(96):
#     if orderedlist[i] == 1:
#         onlyrelevants = np.append(onlyrelevants,[rawnorm[i]])

# onlyrelevants = np.split(onlyrelevants,72)

# fig, ax = plt.subplots(nrows=12, ncols=6,figsize=(32, 20))
# fig.tight_layout(pad=5)
# k=0
# for i in range(12):
#     for j in range(6):
#         ax[i,j].plot(np.linspace(0,rte,length), onlyrelevants[k], '-', markersize = 3)
#         k+=1

# plt.show()


# means = []
# highs = []
# lows = []

# onlyrelevants = np.split(np.array(onlyrelevants),12)


# for i in range(12):
#     if i % 3 == 0:
#         mean = np.mean(np.array([onlyrelevants[i],onlyrelevants[i+1],onlyrelevants[i+2]]),axis=0)
#         std = np.std(np.array([onlyrelevants[i],onlyrelevants[i+1],onlyrelevants[i+2]]),axis=0)
#         means.append(mean)
#         highs.append(mean+std)
#         lows.append(mean-std)
#     else:
#         pass # Odd

means = []
highs = []
lows = []
newonlyrelevants = []
onlyrelevants = np.split(np.array(mostsimilarthrees),4)
for i in onlyrelevants:
    for j in i:
        mean = np.mean(np.array([j[0],j[1],j[2]]),axis=0)
        std = np.std(np.array([j[0],j[1],j[2]]),axis=0)
        means.append(mean)
        highs.append(mean+std)
        lows.append(mean-std)

means = np.split(np.array(means),4)
highs = np.split(np.array(highs),4)
lows = np.split(np.array(lows),4)


# for i in range(4):
#     if i % 3 == 0:
#         mean = np.mean(np.array([onlyrelevants[i],onlyrelevants[i+1],onlyrelevants[i+2]]),axis=0)
#         std = np.std(np.array([onlyrelevants[i],onlyrelevants[i+1],onlyrelevants[i+2]]),axis=0)
#         means.append(mean)
#         highs.append(mean+std)
#         lows.append(mean-std)
#     else:
#         pass # Odd

foldchange = []
foldchangemeans = []
foldchangehighs = []
foldchangelows = []
for i in range(4):
    for j in range(6):
        foldchange.append(np.divide(onlyrelevants[i][j],onlyrelevants[i][5]))
        foldchangemeans.append(np.divide(means[i][j],means[i][5]))
        foldchangehighs.append(np.divide(highs[i][j],highs[i][5]))
        foldchangelows.append(np.divide(lows[i][j],lows[i][5]))




# fig, ax = plt.subplots(nrows=4, ncols=6,figsize=(32, 20))
# fig.tight_layout(pad=5)
# for i in range(4):
#     for j in range(6):
#         ax[i,j].plot(np.linspace(0,rte,length), foldchange[k], '-', markersize = 3)
#         k+=1

# plt.show()


fig, ax = plt.subplots(nrows=4, ncols=5,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
fig.suptitle("Pulse Width Modulated Response",size=26,weight = "bold")
for i in range(4):
    for j in range(6):
        if j == 5:
            k+=1
            break
        if i == 0:
            ymin = 0.7
            ymax = 1.7
            ax[i,j].set_ylim([0.7,1.7])
        elif i == 1:
            ymin = 0.8
            ymax = 1.1
            ax[i,j].set_ylim([0.8,1.1])
        else:
            ymin = 0.7
            ymax = 1.15
            ax[i,j].set_ylim([0.7,1.15])
        ax[i,j].plot(np.linspace(0,rte,length), foldchangemeans[k], '-', linewidth=3)
        ax[i,j].fill_between(np.linspace(0,rte,length),foldchangehighs[k],foldchangelows[k],alpha = 0.45)
        if j == 0:
            ax[i,j].axhspan(ymin, ymax, 0, 1.05/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 1.05/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)
        elif j == 1:
            ax[i,j].axhspan(ymin, ymax, 0, 1.55/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 1.55/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)
        elif j == 2:
            ax[i,j].axhspan(ymin, ymax, 0, 2.05/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 2.05/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)
        elif j == 3:
            ax[i,j].axhspan(ymin, ymax, 0, 2.55/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 2.55/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)
        elif j == 4:
            ax[i,j].axhspan(ymin, ymax, 0, 2.8/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 2.8/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
            ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)


        
        ax[i,j].locator_params(axis='y', nbins=5)
        ax[i,j].locator_params(axis='x', nbins=6)
        ax[i,j].tick_params(axis='both', which='major', labelsize=12)
        if (j!=0):
            if(i==3):
                ax[i,j].tick_params(left = False, right = False , labelleft = False)
            else:
                ax[i,j].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        else:
            if(i==3):
                continue
            else:
                ax[i,j].tick_params( right = False , labelbottom = False, bottom = False)
        k+=1
cols = ['2 hr. pulse', '1.5 hr. pulse', '1 hr. pulse', '30 min. pulse', '15 min. pulse']
rows = ["sucC", "rpoA", "fabA","Anti-sigma\n 28 factor"]

for axi, col in zip(ax[0], cols):
    axi.set_title(col,pad = 20,weight='bold')

for axi, row in zip(ax[:,0], rows):
    axi.set_ylabel(row, rotation=0, size='large',weight='bold',labelpad = 20,ha='right')




fig.supxlabel('Time (hr.)', size='xx-large',weight='bold')
fig.supylabel('Fold Change', size = 22,weight='bold')


plt.show()