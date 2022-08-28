#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/analysis_user/New_trap_code/Tools/')
from basic_packages import * 


def load_dir_reduced(dirname,file_prefix,max_files):
    '''
    Load time information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list1 = []
    var_list2 = []
    var_list3 = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(len(files))
    step_size = 50
    for j in np.arange(0,max_files,step_size):
        for filename in files[j:j+step_size]:
            BDFs = BDF.BeadDataFile(dirname+filename)
            var_list1.append(BDFs.x2)
            var_list2.append(BDFs.cant_pos[1])
            var_list3.append(BDFs.y2)
    return var_list1,var_list2,var_list3


# In[2]:


def data_to_amp_and_phase_single_axis(data,fsamp,res):
    data_det=signal.detrend(data)
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)    # change
    FFT=np.fft.rfft(data_det)
    norm = np.sqrt(2 / (res* fsamp))
    PSD = norm**2 * (FFT * FFT.conj()).real
    Phase =np.angle(FFT)
    return PSD,Phase

## verification of the methods comparability to previously used mlab.psd

# fsamp=5000
# res=50000
# data=f[0].x2
# a = data_to_amp_and_phase_single_axis(data,fsamp,res)
# b = matplotlib.mlab.psd(signal.detrend(data), Fs = fsamp, NFFT = res, window = mlab.window_none)
# plt.loglog(a[0],a[1])
# plt.loglog(b[1],b[0],alpha=0.5)


# In[3]:


def loop_extract_PSD_and_phase(inList,fsamp,res,calibrationFactor=1):
    ampList = []
    phaseList=[]
    for i in range(len(inList)):
        data=inList[i]/calibrationFactor
        temp_ = data_to_amp_and_phase_single_axis(data,fsamp,res)
        ampList.append(temp_[0])
        phaseList.append(temp_[1])
    return ampList,phaseList


# In[4]:


def extract_data_to_df(folderName,filePrefix,maxFiles=1000,filterStd=True,calibrationFactorX=1,calibrationFactorY=1,fsamp=5000,res=50000):
    start=time.time()

    df= pd.DataFrame()
    fsamp=fsamp
    res=res
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)
    x2L,cPL,y2L = load_dir_reduced(folderName,file_prefix=filePrefix,max_files=maxFiles)
    xAmpList,xPhaseList = loop_extract_PSD_and_phase(x2L,fsamp,res,calibrationFactorX)
    yAmpList,yPhaseList = loop_extract_PSD_and_phase(y2L,fsamp,res,calibrationFactorY)
    
    df["cantPosY"] = cPL
    
    df["xAmp"] = x2L
    df["xPhase"] = xPhaseList
    df["xPSD"] = xAmpList
    #df["xASD"]=df["xPSD"].apply(lambda element: np.sqrt(element))
    
    df["yAmp"] = y2L
    df["yPhase"] = yPhaseList
    df["yPSD"] = yAmpList
    #df["yASD"]=df["yPSD"].apply(lambda element: np.sqrt(element))
    
    
    df["checkStd"]=df["xAmp"].apply(lambda element: np.std(element))
    

    if(filterStd==True):
        df=df[df["checkStd"]<20*df["checkStd"].median()]
        df = df.reset_index()
    print("The process took %.2f" %(time.time()-start))
    print(df.info(memory_usage='deep'))

    return freqs,df


# In[5]:


# transform the drive into phases
def add_driveFFT_for_harmonics(df,frequency):
    df["driveFFT"] = df["cantPosY"].apply(lambda element: data_to_amp_and_phase_single_axis(element,fsamp,res)[1][frequency*int(res/fsamp)::frequency*int(res/fsamp)])
    return df


# In[6]:


def plot_compare_amplitudes_tot(frequency,df_1,df_2,method="ASD",axis="y",fsamp=5000,res=50000,
                                       label1="None",label2="None",
                                       lowxlim=10,upxlim=150,lowylim=1e-11,upylim=1,offset=2):
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)

    if method == "Alternate":
        data1=data_to_amp_and_phase_single_axis(df_1["%sAmp" %axis].sum(),fsamp,res)[0]
        data2=data_to_amp_and_phase_single_axis(df_2["%sAmp" %axis].sum(),fsamp,res)[0]            
        xlabel="frequency [Hz]"
        ylabel="%s sumPSD [a.u.]" %axis
    if method == "ASD":
        data1=np.sqrt(df_1["%sPSD" %axis].sum()/len(df_1))
        data2=np.sqrt(df_2["%sPSD" %axis].sum()/len(df_2))
        xlabel="frequency [Hz]"
        ylabel="%s normalized ASD [m/$\sqrt{Hz}$]" %axis
        
        
    plt.plot(freqs,data1,color="black",alpha=1,lw=3,label=label1)
    plt.plot(freqs,data2,color="red",alpha=0.7,lw=3,label=label2)
    plt.yscale("log")
    plt.xlim(lowxlim,upxlim)
    plt.ylim(lowylim,upylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend()
    plt.show()
    
    fundamental=frequency
    plt.plot(freqs[fundamental*int(res/fsamp)::fundamental*int(res/fsamp)],data1[fundamental*int(res/fsamp)::fundamental*int(res/fsamp)],color="black",alpha=1,lw=3,marker="*",ms=15,label=label1)
    plt.plot(freqs[fundamental*int(res/fsamp)::fundamental*int(res/fsamp)],data2[fundamental*int(res/fsamp)::fundamental*int(res/fsamp)],color="red",alpha=0.7,lw=3,marker="*",ms=15,label=label2)
    
    plt.plot(freqs[(fundamental+offset)*int(res/fsamp)::(fundamental+offset)*int(res/fsamp)],data1[(fundamental+offset)*int(res/fsamp)::(fundamental+offset)*int(res/fsamp)],color="blue",alpha=0.7,lw=3,ls="dashed",marker="*",ms=15,label=label1+" off-axis check at %d" %(fundamental+offset))
    plt.plot(freqs[(fundamental+offset)*int(res/fsamp)::(fundamental+offset)*int(res/fsamp)],data2[(fundamental+offset)*int(res/fsamp)::(fundamental+offset)*int(res/fsamp)],color="blue",alpha=0.7,lw=3,ls="dotted",marker="*",ms=15,label=label2+" off-axis check at %d"%(fundamental+offset))

    plt.yscale("log")
    plt.xlim(lowxlim,upxlim)
    plt.ylim(lowylim,upylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# In[7]:


def plot_compare_phases(frequency,df_1,df_2,axis="y",label1="None",label2="None",noHarmonics=10,fsamp=5000,res=50000,compareDrive=False):
    
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)

    
    for j in range(noHarmonics):
        PhaseListY0 = []
        PhaseListY1 = []
        for i in range(len(df_1)):
            PhaseListY0.append(df_1["%sPhase" %axis][i][int(res/fsamp)*frequency*(j+1)]/np.pi)
        for i in range(len(df_2)):
            PhaseListY1.append(df_2["%sPhase" %axis][i][int(res/fsamp)*frequency*(j+1)]/np.pi)
        plt.title("%sth Harmonic" %(j+1))    
        plt.xlabel("%s Phase [$\pi$]" %axis)
        plt.ylabel("normalized Counts")
        plt.hist(PhaseListY0,bins=100,range=(-1,1),label=label1,density=True)
        plt.hist(PhaseListY1,bins=100,range=(-1,1),label=label2,alpha=0.5,density=True)
        if(compareDrive==True):
            plt.axvline(df_1["driveFFT"][0][j]/np.pi,ls="dashed",color="red",lw=7)
        plt.legend()
        plt.show()


# In[8]:


def plot_compare_phases_all(frequency,dfs,axis="y",labels=[],noHarmonics=10,fsamp=5000,res=50000,compareDrive=False):
    
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)

    
    for j in range(noHarmonics):
        k=0
        for df in dfs:
            k+=1
            PhaseListY0 = []
            for i in range(len(df)):
                PhaseListY0.append(df["%sPhase" %axis][i][int(res/fsamp)*frequency*(j+1)]/np.pi)
            plt.hist(PhaseListY0,bins=100,range=(-1,1),label=labels[k-1],density=True,alpha=0.5)
        plt.title("%sth Harmonic" %(j+1))    
        plt.xlabel("%s Phase [$\pi$]" %axis)
        plt.ylabel("normalized Counts") 
        plt.legend()
        plt.show()
        


# In[ ]:




