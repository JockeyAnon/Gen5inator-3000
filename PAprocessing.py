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
length = 170

dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
val = "ExperimentPA"
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
rawnorm = np.split(np.array([i / j for i, j in zip(fluor, od600)]),8)

means=[]
highs = []
lows = []

for i in range(8):
    if i % 2 == 0:
        mean = (np.mean(np.array([rawnorm[i],rawnorm[i+1]]), axis=0))
        std = (np.std(np.array([rawnorm[i],rawnorm[i+1]]), axis=0))
        means.append(mean)
        highs.append(mean+std)
        lows.append(mean-std)
# for i in range(4):
#     mean = np.mean(np.array([rawnorm[i],rawnorm[i+4]]), axis=0)
#     std = np.std(np.array([rawnorm[i],rawnorm[i+4]]), axis=0)
#     means.append(mean)
#     highs.append(mean+std)
#     lows.append(mean-std)
        



foldchange = []
foldchangemeans = [[np.divide(j,i[-1]) for j in i]for i in means]
foldchangehighs = [[np.divide(j,i[-1]) for j in i]for i in highs]
foldchangelows = [[np.divide(j,i[-1]) for j in i]for i in lows]





# fig, ax = plt.subplots(nrows=4, ncols=6,figsize=(32, 20))
# fig.tight_layout(pad=5)
# for i in range(4):
#     for j in range(6):
#         ax[i,j].plot(np.linspace(0,rte,length), foldchange[k], '-', markersize = 3)
#         k+=1

# plt.show()


fig, ax = plt.subplots(nrows=4, ncols=11,figsize=(32, 20))
# fig.tight_layout(pad=8)
plt.subplots_adjust(wspace=0.4, hspace=0.2)
fig.suptitle("Pulse Amplitude Modulated Response",size='xx-large',weight = "bold")
for i in range(4):
    for j in range(11):
        ax[i,j].plot(np.linspace(0,rte,length), foldchangemeans[i][j], '-', markersize = 3)
        ax[i,j].fill_between(np.linspace(0,rte,length),foldchangehighs[i][j],foldchangelows[i][j],alpha = 0.45)
        ymin = 0.85
        if i == 2:
            ymax = 1.25
        else:
            ymax = 1.1

        ax[i,j].axhspan(ymin, ymax, 0, 1.05/7, facecolor=(2/255, 196/255, 73/255), alpha = 0.25)
        ax[i,j].axhspan(ymin, ymax, 1.05/7, 3.1/7, facecolor=(255/255, 54/255, 54/255), alpha = 0.25)
        ax[i,j].axhspan(ymin, ymax, 3.1/7, 1, facecolor=(15/255, 127/255, 255/255), alpha = 0.25)

        ax[i,j].set_ylim([ymin,ymax])

        ax[i,j].locator_params(axis='y', nbins=4)
        ax[i,j].locator_params(axis='x', nbins=6)

cols = []     
for i in range(12):
    cols.append(str(round(10*(0.5**i),3))+" uM")
rows = ["sucC", "rpoA", "fabA","Anti-sigma 28 factor"]

for axi, col in zip(ax[0], cols):
    axi.set_title(col,pad = 10,weight='bold')

for axi, row in zip(ax[:,0], rows):
    axi.set_ylabel(row, rotation=0, size='large',weight='bold',labelpad = 80)




fig.supxlabel('Time (hr.)', size='xx-large',weight='bold')
fig.supylabel('Fold Change', size = 'xx-large',weight='bold')


plt.show()