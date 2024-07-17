
#The list of strain names in array form. The first is formatted so sequential graph generation can take the label of the graph from this array in the same format as the graphs are referenced in subplot array format.
strainlist = [["atpB", "petA", "sucC", "rpoA", "fabA"],["A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA"],["lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]]
#This list of strain names comes in list format so the referencing of graphs is completely sequential in the case of alternative subplot referencing(This is the method used right now.)
strainlist2 = ["atpB", "petA", "sucC", "rpoA", "fabA","A.S. 28 factor", "unchar. protein I", "cspA2", "P. ABC Trans. System", "gitA","lpxC", "unchar. protein II", "capB", "P. O.M. porin A", "acrA"]

#Stores dataframes generated directly from pandas
#   Dataframe array format = [File index(0 = first excel sheet read)][horizontal index(all columns for each row listed in alphabetical and numerical order)][vertical index(time point)]
dataframes = []

#Changes current working directory to the file with the spreadsheet data (REMEMBER TO SET THE PLATE NAMES TO ALPHABETICAL ORDER)
os.chdir(r"C:\Users\Daniel Park\Desktop\Experiment1")
path = os.getcwd() 
f = os.listdir(path)

#Checks for a "processeddata.csv" file that would have formed if the data was previously calculated. This makes it so that the program does not have to run the calculations again just to graph
if os.path.exists(r'C:\Users\Daniel Park\Desktop\Experiment1\processeddata.csv'):
    print("Previous Calculated Data Detected!")
    start_time = time.time()
    #Imports the csv as a data array
    toimport = pd.read_csv("processeddata.csv")
    #Turns the data array into a numpy array, then transposed to align the data to the right lists
    combinedarray = toimport.to_numpy().transpose()
    #Distributes precalculated data into the right variables
    totaldataarray = [literal_eval(i) for i in combinedarray[1]]
    od600dataarray = [literal_eval(i) for i in combinedarray[2]]
    rawnorm1array = [literal_eval(i) for i in combinedarray[3]]
    rawnorm2array = [literal_eval(i) for i in combinedarray[4]]
    raw11 = [literal_eval(i) for i in combinedarray[5]]
    rawcntrlf = [literal_eval(i) for i in combinedarray[6]]
    rawcntrlod = [literal_eval(i) for i in combinedarray[7]]
    rawcntrl = [literal_eval(i) for i in combinedarray[8]]
    print("--- Time elapsed to load data: %s seconds ---" % (time.time() - start_time))
    print("Data Loaded!")
else:
    print("No Previous Calculated Data Detected!")
    start_time = time.time()
    csv_files = glob.glob(os.path.join(path, "*.xlsx")) 

    #Turns all the xlsx spreadsheets into dataframes and transposed to make index calls easier
    for f in csv_files:
        dataframes.append(pd.read_excel(f).values.transpose())
    #Reads the dataframe to know where the OD600 starts
    od600StartIndex = np.where(dataframes[0][0]=="OD:600")[0][0] + 3

    #fold change values are calculated as follows (data/od600)/(control data/control od600)
    def foldchangeval(i,k,l,m,n):
        #ex1. dataframes[i][3+12*k+l][n+m] = referencing on sheet i the value in the cell 4(3+1) + l(the concentration in particular) + 12*k(row number) columns from the left and n+m(dataframe start value + read number)
        #ex2. dataframes[i][3+12*k+11][n+m] = referencing on sheet i the value in the cell 4(3+1) + 11(drags it to the final column value(control)) + 12*k(row number) columns from the left and n+m(dataframe start value + read number) 
        return ((dataframes[i][3+l+12*k][n+m]/dataframes[i][3+l+12*k][od600StartIndex+m])/(dataframes[i][3+11+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m]))
    #Raw Fluorescense data from non control normalized by the corresponding instance's od600
    def rawnorm1(i,k,l,m,n):
        return ((dataframes[i][3+l+(12*k)][n+m]/dataframes[i][3+12*k+l][od600StartIndex+m]))
    #Raw Fluorescense data from control normalized by the corresponding instance's od600
    def rawnorm2(i,k,l,m,n):
        return ((dataframes[i][3+l+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m]))


    #This is for the fold change magnitudes

    #Goes through the dataframes
    for i in range(len(dataframes)):

        for j in range(readProtocols-1):
            n = (od600StartIndex)+(numberOfReads+4)*(j+1)
            for k in range(rowsum):
                for l in range(columnsum):
                    y = []
                    od = []
                    raw1 = []
                    raw2 = []
                    raw = []
                    rawf = []
                    rawod=[]
                    rawcomb=[]
                    for m in range(numberOfReads):
                        if m < timestart:
                            continue
                        if m == timeframe:
                            break
                        y.append(foldchangeval(i,k,l,m,n))
                        od.append(dataframes[i][3+12*k+l][od600StartIndex+m])
                        raw1.append(rawnorm1(i,k,l,m,n))
                        raw2.append(rawnorm2(i,k,l,m,n))
                        raw.append(dataframes[i][3+11+(12*k)][n+m]/dataframes[i][3+11+(12*k)][od600StartIndex+m])
                        rawf.append(dataframes[i][3+12*k+l][n+m])
                        rawod.append(dataframes[i][3+12*k+l][od600StartIndex+m])
                        rawcomb.append(dataframes[i][3+12*k+l][n+m]/dataframes[i][3+12*k+l][od600StartIndex+m])
                    totaldataarray[l].append(y)
                    od600dataarray[l].append(od)
                    rawnorm1array[l].append(raw1)
                    rawnorm2array[l].append(raw2)
                    raw11[l].append(raw)
                    rawcntrlf[l].append(rawf)
                    rawcntrlod[l].append(rawod)
                    rawcntrl[l].append(rawcomb)
    dict = {'total': totaldataarray, 'od600': od600dataarray, 'raw1': rawnorm1array, 'raw2': rawnorm2array, 'raw': raw11, 'rawf': rawcntrlf, 'rawod': rawcntrlod, 'rawcomb': rawcntrl}
    df = pd.DataFrame(dict)
    df.to_csv('processeddata.csv')
    print("--- Time elapsed to calculate data: %s seconds ---" % (time.time() - start_time))
    print('Data file generated!')






newpath = r'C:\Users\Daniel Park\Desktop\Experiment1' 

strainlist22 = ["sucC", "cspA2", "gitA","lpxC"]

tempfig = plt.figure(figsize=(16, 10.0),layout = 'constrained')
tempfig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2,hspace=0, wspace=0)
color = cm.rainbow(np.linspace(0, 1, 12))

for j in range(4):
    hold = tempfig.add_subplot(2,2,j+1)
    hold.set_title(strainlist22[j], fontweight = 'bold',fontsize=20)
    for i, c in enumerate(color):

        areas = np.vstack([totaldataarray[i][0+2*j], totaldataarray[i][1+2*j],totaldataarray[i][2+2*j], totaldataarray[i][3+2*j]])
        means = areas.mean(axis=0)
        stds = areas.std(axis=0, ddof=1)
        if i == 11:
            hold.plot(np.linspace(0,217,217), means, '-', c=c, label='C = 0 uM', linewidth=2)

        else:
            hold.plot(np.linspace(0,217,217), means, '-', markersize = 1, c=c, label='C = ' + str(round(10*(0.5**i),3)) + ' uM', linewidth=1.0)

        hold.fill_between(np.linspace(0,217,217), means - stds, means + stds,color=c, alpha=0.4)
        hold.set_ylim([0.6, 1.2])
        hold.set_xticks(np.linspace(0, 217, 217),fontweight='bold')
        hold.set_yticks(np.linspace(0.6,1.2,10),fontweight='bold')
        hold.set_xticklabels([int(number) for number in (np.linspace(0,18,217))],fontweight='bold')
        hold.set_yticklabels(np.around(np.arange(0.6,1.2,0.06),2),fontweight='bold')
        hold.tick_params(axis='both', labelsize=20)
        hold.locator_params(nbins = 3, axis='both')
        hold.spines[['right', 'top']].set_visible(False)
        
filename = 'C:/Users/Daniel Park/Desktop/NewDataSet/Fidelities/Fixed Concentration, Time/' + 'overlapped concentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
tempfig.savefig(filename)
plt.close(tempfig)




# This is array form of averaged with error bands, fixed time, concentration dependent


strainlist22 = ["sucC", "cspA2", "gitA","lpxC"]

for i in range(numberOfReads):
    a = 0
    b = 0
    
    if i == 0 or i == 50 or i == 70 or i == 100 or i == 140 or i == 200:
        fig, ax = plt.subplots(2,2,figsize=(24, 12),layout = 'constrained')
        fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.2,hspace=0, wspace=0)
        #fig.suptitle('Fold-Change Magnitudes At ' + str(i*5) + ' Minutes', fontsize=25)
        #fig.supylabel('Fold Change Magnitude', fontsize=25)
        #fig.supxlabel('Concentration (uM)', fontsize=25)
        
        for j in range(4):
            cmean = []
            clow = []
            chigh = []
            for k in range(12):
                areas = np.vstack([totaldataarray[k][0+2*j][i] , totaldataarray[k][1+2*j][i], totaldataarray[k][2+2*j][i] , totaldataarray[k][3+2*j][i]])
                means = areas.mean(axis=0)
                stds = areas.std(axis=0, ddof=1)
                cmean.append(means[0])
                clow.append(means[0]-stds[0])
                chigh.append(means[0]+stds[0])
            ax[a,b].set_title(strainlist22[j], fontsize=30,pad=35, weight = 'bold')
            ax[a,b].semilogx(np.logspace(11, 0, 12, base = 0.5), cmean, '.', markersize = 25,color = 'blue', base = 0.5)
            ax[a,b].fill_between(np.logspace(11, 0, 12, base = 0.5), clow, chigh, alpha=0.7)
            ax[a,b].set_ylim([0, 1.5])
            ax[a,b].spines[['right', 'top']].set_visible(False)
            ax[a,b].set_xscale('log', base = 0.5)
            ax[a,b].xaxis.set_major_formatter(ScalarFormatter())
            ax[a,b].minorticks_off()
            ax[a,b].set_xticks(np.logspace(11, 0, 12, base = 0.5),fontweight='bold')
            ax[a,b].set_yticks(np.linspace(0.6,1.0,10),fontweight='bold')
            #ax[a,b].set_xticklabels([['10',r'0.5$\bf{^1}$',r'0.5$\bf{^2}$',r'0.5$\bf{^3}$',r'0.5$\bf{^4}$',r'0.5$\bf{^5}$',r'0.5$\bf{^6}$',r'0.5$\bf{^7}$',r'0.5$\bf{^8}$',r'0.5$\bf{^9}$',r'0.5$\bf{^10}$','0',]],fontweight='bold')
            ax[a,b].set_xticklabels([str(round(10*(0.5**i),2)) for i in range(12)],fontweight='bold')
            
            ax[a,b].set_yticklabels(np.arange(0.6,1.0,0.04),fontweight='bold')
            # ax[a,b].tick_params(axis='both', labelsize=20)
            ax[a,b].locator_params(nbins = 2, axis='both')
            ax[a,b].tick_params(axis='both', labelsize=30)
            
            b = b+1
            
            if b == 2:
                b = 0
                a = a+1
        filename = 'C:/Users/Daniel Park/Desktop/NewDataset/Fidelities/Fixed Time, Concentration/' + 'time' + str(i) + ' grid.png'
        fig.savefig(filename)
        plt.close(fig)

# This is for first run, fixed concentration, time dependent
i = 0
j = 3
tempfig = plt.figure()
hold = tempfig.add_subplot()
hold.plot(np.linspace(0,217,217), rawcntrlf[i][0+2*j], '-', markersize = 3,color = 'blue')
hold.plot(np.linspace(0,217,217), rawcntrlf[i][1+2*j], '-', markersize = 3,color = 'red')
hold.plot(np.linspace(0,217,217), rawcntrlf[i][2+2*j], '-', markersize = 3,color = 'green')
hold.plot(np.linspace(0,217,217), rawcntrlf[i][3+2*j], '-', markersize = 3,color = 'orange')
filename = 'C:/Users/Daniel Park/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolfconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
tempfig.savefig(filename)
plt.close(tempfig)

i = 0
j = 3
tempfig = plt.figure()
hold = tempfig.add_subplot()
hold.plot(np.linspace(0,217,217), rawcntrlod[i][0+2*j], '-', markersize = 3,color = 'blue')
hold.plot(np.linspace(0,217,217), rawcntrlod[i][1+2*j], '-', markersize = 3,color = 'red')
hold.plot(np.linspace(0,217,217), rawcntrlod[i][2+2*j], '-', markersize = 3,color = 'green')
hold.plot(np.linspace(0,217,217), rawcntrlod[i][3+2*j], '-', markersize = 3,color = 'orange')
filename = 'C:/Users/Daniel Park/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolodconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
tempfig.savefig(filename)
plt.close(tempfig)

i = 0
j = 3
tempfig = plt.figure()
hold = tempfig.add_subplot()
hold.plot(np.linspace(0,217,217), rawcntrl[i][0+2*j], '-', markersize = 3,color = 'blue')
hold.plot(np.linspace(0,217,217), rawcntrl[i][1+2*j], '-', markersize = 3,color = 'red')
hold.plot(np.linspace(0,217,217), rawcntrl[i][2+2*j], '-', markersize = 3,color = 'green')
hold.plot(np.linspace(0,217,217), rawcntrl[i][3+2*j], '-', markersize = 3,color = 'orange')
filename = 'C:/Users/Daniel Park/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusoncntrolconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
tempfig.savefig(filename)
plt.close(tempfig)

i = 0
j = 3
tempfig = plt.figure()
hold = tempfig.add_subplot()
hold.plot(np.linspace(0,217,217), totaldataarray[i][0+2*j], '-', markersize = 3,color = 'blue')
hold.plot(np.linspace(0,217,217), totaldataarray[i][1+2*j], '-', markersize = 3,color = 'red')
hold.plot(np.linspace(0,217,217), totaldataarray[i][2+2*j], '-', markersize = 3,color = 'green')
hold.plot(np.linspace(0,217,217), totaldataarray[i][3+2*j], '-', markersize = 3,color = 'orange')
filename = 'C:/Users/Daniel Park/Desktop/NewDataSet/First Runs/Fixed Concentration, Time/' + 'focusonfoldchangeconcentration 0.5^' + str(i) + ' strain ' + str(j+1) + '.png'
tempfig.savefig(filename)
plt.close(tempfig)



                