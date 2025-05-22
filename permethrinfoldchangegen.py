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
import scipy
from matplotlib import pyplot
import statistics as st
import seaborn as sns
mpl.use('TkAgg')

rte = 18
length = 216

dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
val = "PermethrinFinal"
newdir = "/Users/danielpark/Desktop/"+val+"/"
os.chdir(newdir)
path = os.getcwd() 
f = os.listdir(path)


print("No Previous Calculated Data Detected!")
start_time = time.time()
csv_files = glob.glob(os.path.join(path, "*.xlsx")) 


for f in csv_files:
    dataframes.append(pd.read_excel(f).values.transpose())

od600 = dataframes[0]
fluor = dataframes[1]
fluor2 = dataframes[2]
rawnorm = [i / j for i, j in zip(fluor, od600)]
rawnorm = [np.split(np.array(i),8) for i in np.split(np.array(rawnorm),4)]


onlyrelevants = []
means = []
highs = []
lows = []

for i in rawnorm:
    for j in i:
        rowdata = j.tolist()
        control = j[11]
        onlyrelevants.append([np.divide(np.array(k),np.array(control)) for k in rowdata])
onlyrelevants = np.split(np.array(onlyrelevants).flatten(),384)
onlyrelevants = [np.split(np.array(i),8) for i in np.split(np.array(onlyrelevants),4)]
means = []
stds = []
for i in onlyrelevants:
    for j in range(8):
        if j % 2 == 0:
            means.append(np.mean(np.array([i[j],i[j+1]]),axis=0))
            stds.append(np.std(np.array([i[j],i[j+1]]),axis=0))
means = np.split(np.array(means).flatten(),192)
stds = np.split(np.array(stds).flatten(),192)
yhat = scipy.signal.savgol_filter(means, 30, 3)
xhat = scipy.signal.savgol_filter(stds, 30, 3)
highs = yhat + xhat
lows = yhat - xhat



fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(32, 10))

plt.subplots_adjust(wspace=0.2, hspace=0.3)
l = 0
for i in range(3):
    for j in range(5):
        for k in range(12):
            if k == 0:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'blue' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'blue', errorevery = (0,25))
            if k == 2:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'cyan' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'cyan', errorevery = (0,25))
            elif k == 4:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'purple' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'purple',errorevery = (0,25))
            elif k == 6:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'green' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'green', errorevery = (0,25))
            elif k == 8:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'orange' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'orange', errorevery = (0,25))
            elif k == 10:
                ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = 'red' )
                # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
                ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = 'red', errorevery = (0,25))
            # ax[i, j].set_xlim(0, 18)
            # ax[i, j].set_ylim(0.8, 1.25)
            ax[i,j].locator_params(axis='y', nbins=4)
            ax[i,j].locator_params(axis='x', nbins=3)
            
            l+=1
fig.supxlabel('Time (hr.)', size='xx-large',weight='bold', position = (0.5,0.05))
fig.supylabel('Fold Change', size = 'xx-large',weight='bold',position = (0.075,0.5))

fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
l = 0
m=0
color = cm.rainbow(np.linspace(0, 1, 12))
strainlist2 = ["atpB", "petA", "sucC", "rpoA", "fabA","A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA","lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]

for i in range(3):
    for j in range(5):
        for k, c in enumerate(color):
            ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 5, color = c )
            # ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.4**k))
            ax[i,j].errorbar(np.linspace(0,rte,length),yhat[l],stds[l], color = c, errorevery = (0,25))
            ax[i,j].locator_params(axis='y', nbins=4)
            ax[i,j].locator_params(axis='x', nbins=3)
            ax[i,j].title.set_text(strainlist2[m])
            l+=1
        m+=1

fig.supxlabel('Time (hr.)', size='xx-large',weight='bold')
fig.supylabel('Fold Change', size = 'xx-large',weight='bold')


plt.show()

fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
l = 0
for i in range(3):
    for j in range(5):
        for k in range(12):
            ax[i,j].plot(np.linspace(0,rte,length), yhat[l], '-', linewidth = 4, color = 'blue' ,alpha = (0.7**k))
            ax[i,j].fill_between(np.linspace(0,rte,length), lows[l], highs[l] ,color = 'blue' ,alpha = (0.1))
            ax[i,j].locator_params(axis='y', nbins=4)
            ax[i,j].locator_params(axis='x', nbins=3)
            l+=1





fig.supxlabel('Time (hr.)', size='xx-large',weight='bold')
fig.supylabel('Fold Change', size = 'xx-large',weight='bold')


plt.show()

yhat = yhat.transpose()
highs = highs.transpose()
lows = lows.transpose()


strainlist2 = ["atpB", "petA", "sucC", "rpoA", "fabA","A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA","lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]


fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
l = 0
labels = []
for i in range(12):
    labels.append(round((20*(0.5**i)),2))

for i in range(3):
    for j in range(5):
        
        # ax[i,j].plot(np.linspace(0,11,6), yhat[96][(12*l):(11+12*l):2], '.', markersize = 8, color = 'blue' )
        ax[i,j].plot(np.linspace(0,11,6), yhat[96][(12*l):(11+12*l):2], '-', linewidth = 2, color = 'gray' )
        
        ax[i,j].plot(0, yhat[96][0+12*l], '.', markersize = 15, color = 'purple' ,zorder = 2)
        ax[i,j].plot(2.15, yhat[96][2+12*l], '.', markersize = 15, color = 'blue' ,zorder = 2)
        ax[i,j].plot(4.35, yhat[96][4+12*l], '.', markersize = 15, color = 'green' ,zorder = 2)
        ax[i,j].plot(6.575, yhat[96][6+12*l], '.', markersize = 15, color = 'yellow' ,zorder = 2)
        ax[i,j].plot(8.8, yhat[96][8+12*l], '.', markersize = 15, color = 'orange' ,zorder = 2)
        ax[i,j].plot(11, yhat[96][10+12*l], '.', markersize = 15, color = 'red' ,zorder = 2)
        ax[i,j].errorbar(np.linspace(0,11,6),yhat[96][(12*l):(11+12*l):2],stds[96][(12*l):(11+12*l):2], color='blue',zorder = 1)
        ax[i,j].locator_params(axis='y', nbins=4)
        # ax[i,j].locator_params(axis='x', nbins=3)
        ax[i,j].title.set_text(strainlist2[l])
        ax[i,j].set_xticklabels(labels)
        l+=1




fig.supxlabel('Concentration (uM)', size='xx-large',weight='bold',y=1)
fig.supylabel('Fold Change', size = 'xx-large',weight='bold',x=1)


plt.show()