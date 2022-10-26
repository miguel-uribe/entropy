import numpy as np
import pandas as pd
import antropy as ent
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq 
import seaborn as sns
from scipy import signal
import seaborn as sns
from scipy import stats
from os.path import exists
import sys

from definitions import *

# this script should be called entropy_calculations subject phase1 phase2

if __name__ == '__main__':
    
    subject = sys.argv[1]
    phases = [sys.argv[2],sys.argv[3]]
    
    sesiones = np.arange(1,20, dtype = int)
    
    datosT = []
    
    for ses in sesiones:
        datos = {}
        for phase in phases:
            try:
                datos[phase] = pd.read_csv('../data/'+subject+'/'+subject+'_'+phase+'_%d.txt'%ses, index_col=False, header=6)
            except:
                #print("Failed to load "+ '../data/'+subject+'/'+subject+'_'+phase+'_%d.txt'%ses)
                datos[phase] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
                 
        datosT.append(datos)
        #print(datosT)

    #for sample and permutation entropies
    for metric in np.arange(2,20,2):
        getEntropyAnalysis(subject, phases, datosT,'sampen', sesiones, metric, 1, 240, 1, 0.8)
    for metric in np.arange(3,20,2):
        getEntropyAnalysis(subject, phases, datosT,'permen', sesiones, metric, 1, 240, 1, 0.8)
        
    # now for the multiescale entropy
    for scale in np.arange(2,20,2):
        getEntropyAnalysis(subject, phases, datosT,'msen', sesiones, 2, scale, 240, 1, 0.8)
        getEntropyAnalysis(subject, phases, datosT,'msen', sesiones, 3, scale, 240, 1, 0.8)
