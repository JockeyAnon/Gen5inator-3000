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
import time


startTOTAL_time = time.time()
#Initial Declaration for the processed data cache. Each corresponds to a concentration, while the last one corresponds to the control. There are 12 arrays nested within each array for each graph type.
#The reason why the data is stored in cache type instead of processed on-demand is because the final data graphs use the same data but in different visualizations.

totaldataarray = []

raw11 = []
od600dataONLY = []
fluorescencedataONLY = []


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

#This list of strain names comes in list format so the referencing of graphs is completely sequential in the case of alternative subplot referencing(This is the method used right now.)
strainlist2 = ["sucC", "rpoA", "fabA","A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA"]

inducerlist = ["Calcium Hydroxide", "Tryptophan", "Hemin", "Crude Oil", "Naringenin",	"Acrylic Acid",	"L Cysteine",	"D Mannitol",	"Phenylalanine",	"Glycine",	"a ketoglutaric acid",	"d xylose","bile salts",	"citric acid",	"p alanine",	"glutamic acid",	"trycine",	"MgCl",	"tween80",	"pectin",	"Erythromycin",	"Paraffin",	"Casein",	"TPA" ,"Succinic Acid",	"SSDH",	"Tryptone",	"Malic Acid",	"Vanillic Acid",	"Sodium Bicarbonate",	"L Arabinose",	"Rhamnose MH",	"Starch",	"L proline",	"Mucin",	"L valine","fructose",	"Ferric Ammonium CItrate",	"Guanosine",	"A. Galactan",	"Dextran",	"L TMEH",	"MOPS",	"L APS",	"L Threonine",	"ZnCl",	"CaCl2 Anhydrate",	"Inulin"]

#Stores dataframes generated directly from pandas
#   Dataframe array format = [File index(0 = first excel sheet read)][horizontal index(all columns for each row listed in alphabetical and numerical order)][vertical index(time point)]
dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
val = "Experiment8"
newdir = "C:/Users/Daniel Park/Desktop/"+val+"/"
os.chdir(newdir)
path = os.getcwd() 
f = os.listdir(path)





start_time = time.time()
csv_files = glob.glob(os.path.join(path, "*.xlsx")) 

#Turns all the xlsx spreadsheets into dataframes and transposed to make index calls easier
for f in csv_files:
    transposeFrame = pd.read_excel(f).values.transpose()
    dataframes.append(transposeFrame)
    od600dataONLY.append([z[40:257] for z in transposeFrame[3:99]])
    fluorescencedataONLY.append([z[261:478] for z in transposeFrame[3:99]])


raw = [np.divide(fl,o) for fl,o in zip(fluorescencedataONLY,od600dataONLY)]

for b in range(12):
    totaldataarray.append(np.divide(raw[0][b::12],raw[1][1::12]))



newpath1 = newdir + "Graphics/"     
if not os.path.exists(newpath1):
    os.makedirs(newpath1)

# # This is for averaged with error bands with concentrations overlapped
# print("Generating Graph Set 1...")
# start_time = time.time()


# for i in range(8):
#     b=i*12
#     a=0
#     tempfig = plt.figure(figsize=(24, 15.0),layout = 'constrained')
#     tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)
#     for j in range(48):
#         hold = tempfig.add_subplot(6,8,j+1)
#         hold.set_title(inducerlist[j], fontweight = 'bold',fontsize=20)
#         hold.plot(np.linspace(0,217,217), totaldataarray[a][b], '-', linewidth=2)
#         b=b+1
#         if b == 12+i*12:
#             a=a+1
#             b=i*12


#     newpath = newpath1 + "Strains/Fixed Strain, InducerArray/" 
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)           
#     filename = newpath + 'strain' + str(i+1) + '.png'
#     tempfig.savefig(filename)
#     plt.close(tempfig)
# print("--- Time elapsed to generate this graph set: %s seconds ---" % (time.time() - start_time))
# print("Graphs Generated and Stored!")


print("Generating Graph Set 2...")
start_time = time.time()
covorder = []
covlistcomp = []
covliststdcomp = []

covlist = []
covstdlist = []

tempfig = plt.figure(figsize=(24, 15.0),layout = 'constrained')
tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)

for i in range(4):
    for j in range(12):
        areas = np.vstack([np.vectorize(float)(totaldataarray[j][i]), np.vectorize(float)(totaldataarray[j][i+4])])
        means = areas.mean(axis=0)
        stds = areas.std(axis=0)
        covlist.append(means)
        covstdlist.append(stds)


covariancematix = np.cov(np.vstack(covlist))
plt.matshow(covariancematix)
plt.show()
covorder.append(np.flip(np.argsort(covariancematix[-1])))

covlistcomp.append(covlist)
covliststdcomp.append(covstdlist)

for i in range(1):
    tempfig = plt.figure(figsize=(24, 15.0),layout = 'constrained')
    tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)
    for j in range(48):
        hold = tempfig.add_subplot(6,8,j+1)
        hold.set_title(inducerlist[covorder[i][j]], fontweight = 'bold',fontsize=20)
        hold.plot(np.linspace(0,217,217), covlistcomp[i][covorder[i][j]], '-', linewidth=2)
        hold.fill_between(np.linspace(0,217,217), covlistcomp[i][covorder[i][j]] - covliststdcomp[i][covorder[i][j]], covlistcomp[i][covorder[i][j]] + covliststdcomp[i][covorder[i][j]], alpha=0.4)


    newpath = newpath1 + "Strains COV/Fixed Strain, InducerArray/" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)           
    filename = newpath + 'strain' + str(i+1) + '.png'
    tempfig.savefig(filename)
    plt.close(tempfig)

# for i in range(8):
#     plt.figure(i)
#     ax = plt.axes(projection='3d')
#     xlist = []
#     ylist = []
#     zlist = []
#     labels = []
#     for j in range(46):
#         xlist.append(np.linspace(0,217,217))
#         ylist.append(np.full((timeframe-timestart, ), j+2))
#         zlist.append(covlistcomp[i][covorder[i][j+2]])
#         labels.append(inducerlist[covorder[i][j+2]])
#     ax.set_title("Strain " + strainlist2[i])
#     ax.plot_surface(np.array(xlist), np.array(ylist), np.array(zlist),cmap=cm.coolwarm,antialiased=True)
#     ax.set_xlim3d(0, timeframe-timestart)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Inducer')
#     ax.set_zlabel('Fold Change')
# plt.show()