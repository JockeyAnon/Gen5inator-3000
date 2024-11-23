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
mpl.use('Agg')

startTOTAL_time = time.time()
#Initial Declaration for the processed data cache. Each corresponds to a concentration, while the last one corresponds to the control. There are 12 arrays nested within each array for each graph type.
#The reason why the data is stored in cache type instead of processed on-demand is because the final data graphs use the same data but in different visualizations.


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

graphcounter = 1

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
    # dataframes.append(pd.read_excel(f).values)

od600dataarraywc = np.array_split(np.append(od600dataarraywc,dataframes[0]),96)
fluordataarraywc = np.array_split(np.append(fluordataarraywc,dataframes[1]),96)
controlod600 = od600dataarraywc[12 - 1::12]
controlfluor = fluordataarraywc[12 - 1::12]
rawnorm = np.divide(fluordataarraywc,od600dataarraywc)
rawnormwc = np.divide(fluordataarraywc,od600dataarraywc)
rawnormc = np.divide(controlfluor,controlod600)

od600dataarray = np.array_split(od600dataarraywc, 8)
od600dataarraywc = np.array_split(od600dataarraywc, 8)
fluordataarray = np.array_split(fluordataarraywc, 8)
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



print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - start_time))


newpath1 = newdir + "Graphics/"     
if not os.path.exists(newpath1):
    os.makedirs(newpath1)
length = len(fc1[0][0])

def var_name(var):
    for name, value in globals().items():
        if value is var:
            return name



def avgstdpair(source):
    return np.array_split(np.vstack(([np.mean(i,axis=0) for i in np.array_split(source, 4)], [np.std(i,axis=0) for i in np.array_split(source, 4)])),2)
def avgsqrstdpair(source):
    return np.array_split(np.vstack(([np.mean(i,axis=0) for i in np.array_split([np.mean(i,axis=0) for i in np.array_split(source, 4)], 4)], [np.std(i,axis=0) for i in np.array_split([np.mean(i,axis=0) for i in np.array_split(source, 4)], 4)]) ),2)

def graph(source,type,averaged,errors,lod,smoothing,fou,rte,graphcounter):
    print("Generating Graph Set " + str(graphcounter) + "...")
    start_time = time.time()
    newpath = newpath1 + str(var_name(source)) + "/Fixed Concentration, Time/"
    plt.rcParams['font.weight'] = 'bold'

    if averaged == 0:
        newpath+="unaveraged/"
        zeros = np.zeros(np.array(source).shape)
        source = np.vstack((source,zeros))
        source = np.array_split(source,2)
    elif averaged == 1:
        newpath+="averaged by row/"
        source = avgstdpair(source)
    elif averaged == 2:
        newpath+="averaged by squares/"
        source = avgsqrstdpair(source)
    else:
        return

    if errors == 0:
        newpath+="no errors/"
    elif errors == 1:
        newpath+="with errors/"
        highs = np.add(source[0],source[1])
        lows = np.subtract(source[0],source[1])
        
    else:
        return

    dim1=len(source[0])
    dim2=len(source[0][0])
    color = cm.rainbow(np.linspace(0, 1, dim2))
    
    unittype = ''
    if lod == 0:
        unittype = '.'
        newpath+="dots/"
    elif lod == 1:
        unittype = '-'
        newpath+="lines/"
    else:
        return    

    if smoothing == 0:
        source = source
        newpath+="not smoothed/"
    elif smoothing == 1:
        source =[signal.savgol_filter(x,50,1) for x in [y for y in source]]
        highs = np.add(source[0],source[1])
        lows = np.subtract(source[0],source[1])
        newpath+="smoothed/"
    else:
        return
    
    if fou == 0:
        newpath+="unfixed axes/"
    elif fou == 1: 
        newpath+="fixed axes/"
        minval = np.max(np.concatenate(source[0]).ravel())
        maxval = np.min(np.concatenate(source[0]).ravel())
    else:
        return


    if type == 0:
        fig, ax = plt.subplots(nrows=dim1, ncols=dim2,figsize=(32, 20))
        fig.tight_layout(pad=5)
        newpath+="array/"
    if type == 1:     
            newpath+="stacked/" 
    if type == 2:
                newpath+="singles/"
    for i in range(dim1):
        if type == 1:
            tempfig = plt.figure(figsize=(32, 20))
        for j , c in enumerate(color):   
            if type == 2:
                tempfig = plt.figure(figsize=(32, 20))
            if type == 1 or type == 2:
                plt.plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                if errors == 1:
                    plt.fill_between(np.linspace(0,rte,length), highs[i][j], lows[i][j],  color = c, alpha = 0.5)
                if fou == 1:
                    plt.ylim([maxval,minval])
            elif type == 0:
                ax[i,j].plot(np.linspace(0,rte,length), source[0][i][j], unittype, markersize = 3, color = c)
                if errors == 1:
                    ax[i,j].fill_between(np.linspace(0,rte,length), highs[i][j], lows[i][j],  color = c, alpha = 0.5)
                if fou == 1:
                    ax[i,j].set_ylim([maxval,minval])
            else:
                return
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
    print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
    print("Graphs Generated and Stored!")
    
print("Generating all graph sets for current file ...")
start_timeoverall = time.time()
for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        graph(od600dataarraywc,i,j,k,l,m,n,5,graphcounter)
                        graph(fluordataarraywc,i,j,k,l,m,n,5,graphcounter)
                        graph(fc1,i,j,k,l,m,n,5,graphcounter)
                        graph(rawnormwc,i,j,k,l,m,n,5,graphcounter)
                        graphcounter+=1

# while True:
#     averages = input("Enter what kind of graph you want")


print("--- Time elapsed to generate all graphs: %s seconds ---" % (time.time() - start_time))