import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mticker
import scipy
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
val = "PermethrinFinal"
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
fluortight = dataframes[1]
fluorwide = dataframes[2]
rawnorm1 = [i / j for i, j in zip(fluortight, od600)]
rawnorm2 = [i / j for i, j in zip(fluorwide, od600)]

rawnorm1split = np.split(np.array(rawnorm1),16)
rawnorm2split = np.split(np.array(rawnorm2),16)

rawnormself1normed = []
rawnormself2normed = []


for i in range(16):
    rawnorm1split = np.split(np.array(rawnorm1),16)
    rawnorm2split = np.split(np.array(rawnorm2),16)
    focus1 = rawnorm1split.pop(i)
    focus2 = rawnorm2split.pop(i)
    
    rawnormself1normed.append([[k/j for k,j in zip(i,focus1)] for i in rawnorm1split])
    rawnormself2normed.append([[k/j for k,j in zip(i,focus2)] for i in rawnorm2split])

averagedselfnormed1 = []
averagedselfnormed2 = []

for i in rawnormself1normed:
    temphold1 = []
    for j in i:
        temphold2 = []
        for k in range(24):
            if k % 2 == 0:
                temphold2.append(np.mean(np.array([j[k],j[k+1]]),axis=0))
            else:
                pass
        temphold1.append(temphold2)
    averagedselfnormed1.append(temphold1)

for i in rawnormself2normed:
    temphold1 = []
    for j in i:
        temphold2 = []
        for k in range(24):
            if k % 2 == 0:
                temphold2.append(np.mean(np.array([j[k],j[k+1]]),axis=0))
            else:
                pass
        temphold1.append(temphold2)
    averagedselfnormed2.append(temphold1)


        
          

newpath1 = newdir + "Graphics1/"     
newpath = newpath1 + "/Fixed Concentration, Time/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(16):
    for j in range(15):
        fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(32, 20))
        fig.tight_layout(pad=5)
        m=0
        for k in range(2):
            for l in range(6):
                ax[k,l].plot(np.linspace(0,18,216), signal.savgol_filter(averagedselfnormed1[i][j][m],27,1), "-", markersize = 3, color = 'blue')
                m+=1
        filename = newpath + "strain " + str(i+1) + " normalized by strain " + str(j+1)
        fig.savefig(filename)
        plt.close(fig)


newpath1 = newdir + "Graphics2/"     
newpath = newpath1 + "/Fixed Concentration, Time/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(16):
    for j in range(15):
        fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(32, 20))
        fig.tight_layout(pad=5)
        m=0
        for k in range(2):
            for l in range(6):
                ax[k,l].plot(np.linspace(0,18,216), signal.savgol_filter(averagedselfnormed2[i][j][m],27,1), "-", markersize = 3, color = 'blue')
                m+=1
        filename = newpath + "strain " + str(i+1) + " normalized by strain " + str(j+1)
        fig.savefig(filename)
        plt.close(fig)







print("Done")