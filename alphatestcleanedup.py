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

rte = 5
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

rawnorm = [i / j for i, j in zip(fluor, od600)]


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






fig, ax = plt.subplots(nrows=4, ncols=5,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ymin = 0.7
ymax=0
fig.suptitle("Pulse Width Modulated Response",size='xx-large',weight = "bold")
for i in range(4):
    for j in range(6):
        if j == 5:
            k+=1
            break
        if i == 0:
            ymax = 1.7
        elif i == 1:
            ymax = 1.3
        else:
            ymax=1.2
        ax[i,j].set_ylim([ymin,ymax])
        ax[i,j].plot(np.linspace(0,rte,length), foldchangemeans[k], '-', markersize = 3)
        ax[i,j].fill_between(np.linspace(0,rte,length),foldchangehighs[k],foldchangelows[k],alpha = 0.45)
        ax[i,j].locator_params(axis='y', nbins=4)
        ax[i,j].locator_params(axis='x', nbins=3)
        ax[i,j].tick_params(axis='both', which='major', labelsize=12)
        ax[i,j].axhspan(ymin, ymax, 0, 1.05/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
        ax[i,j].axhspan(ymin, ymax, 1.05/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
        ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)
        k+=1
cols = ['2 hr. pulse', '1.5 hr. pulse', '1 hr. pulse', '30 min. pulse', '15 min. pulse']
rows = ["sucC", "rpoA", "fabA","A.S. 28 factor"]

for axi, col in zip(ax[0], cols):
    axi.set_title(col,pad = 10,weight='bold')

for axi, row in zip(ax[:,0], rows):
    axi.set_ylabel(row, rotation=0, size='large',weight='bold',labelpad = 40)




fig.supxlabel('Time (hr.)', size='xx-large',weight='bold')
fig.supylabel('Fold Change', size = 'xx-large',weight='bold')


plt.show()