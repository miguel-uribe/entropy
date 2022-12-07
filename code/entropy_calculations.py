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

# this script should be called entropy_calculations subject phase1 phase2 threshold

if __name__ == '__main__':
    
    subject = sys.argv[1]
    phases = [sys.argv[2],sys.argv[3]]
    threshold = float(sys.argv[4])
    
    sesiones = np.arange(1,20, dtype = int)
    
    datosT = []
    
    for ses in sesiones:
        datos = {}
        for phase in phases:
            try:
                datos[phase] = pd.read_csv('../data/'+subject+'/'+subject+'_'+phase+'_%d.txt'%ses, index_col=False, header=6)
                print("Reading Session: %d, Phase: %s"%(ses,phase))
            except:
                print("Failed to read Session: %d, Phase: %s"%(ses,phase))
                datos[phase] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
                 
        #print(datos)
        datosT.append(datos)
        #print(datosT)

    #print(datosT)
    
    #for sample and permutation entropies
    for metric in np.arange(2,20,2):  #20
        calculateEntropy(subject, phases, datosT,'sampen', sesiones, metric, 1, 240, 0.4, 0.8, threshold)
        getEntropyEvolution(subject, phases, 'sampen', metric, 1, 0.4, 0.8)
    for metric in np.arange(3,20,2):
        calculateEntropy(subject, phases, datosT,'permen', sesiones, metric, 1, 240, 0.4, 0.8, threshold)
        getEntropyEvolution(subject, phases, 'permen', metric, 1, 0.4, 0.8)

        
    # now for the multiescale entropy
    for scale in np.arange(2,20,2):
        calculateEntropy(subject, phases, datosT,'msen', sesiones, 2, scale, 240, 0.4, 0.8, threshold)
        getEntropyEvolution(subject, phases, 'msen', 2, scale, 0.4, 0.8)
        calculateEntropy(subject, phases, datosT,'msen', sesiones, 3, scale, 240, 0.4, 0.8, threshold)
        getEntropyEvolution(subject, phases, 'msen', 3, scale, 0.4, 0.8)
