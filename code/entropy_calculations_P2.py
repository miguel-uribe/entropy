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

from definitions import *

# this script should be called 

if __name__ == '__main__':
    


"""
datosC = []
controles = np.array([1,2,3])

for con in controles:
    datos = {}
    # inicial
    try:
        datos['inicial'] = pd.read_csv('../data/Control/C%d_inicial.txt'%con, index_col=False, header=6)
    except:
        datos['inicial'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
    # final
    try:
        datos['final'] = pd.read_csv('../data/Control/C%d_final.txt'%con, index_col=False, header=6)
    except:
        datos['final'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    datosC.append(datos)

# for sample and permutation entropies
for metric in np.arange(2,20,2):
        getEntropyAnalysis('control', ['inicial', 'final'], datosC,'sampen', controles, metric, 1, 240, 1, 0.8)
for metric in np.arange(3,20,2):
        getEntropyAnalysis('control', ['inicial', 'final'], datosC,'permen', controles, metric, 1, 240, 1, 0.8)
        
# now for the multiescale entropy
for scale in np.arange(2,20,2):
        getEntropyAnalysis('control', ['inicial', 'final'], datosC,'msen', controles, 2, scale, 240, 1, 0.8)
        getEntropyAnalysis('control', ['inicial', 'final'], datosC,'msen', controles, 3, scale, 240, 1, 0.8)


datosP1 = []
sesiones1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

for ses in sesiones1:
    datos = {}
    # presesion
    try:
        datos['pre'] = pd.read_csv('../data/P1/P1_pre_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['pre'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
    # sesion
    try:
        datos['sesion'] = pd.read_csv('../data/P1/P1_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['sesion'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    try:
        datos['post'] = pd.read_csv('../data/P1/P1_post_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['post'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    datosP1.append(datos)


# for sample and permutation entropies
for metric in np.arange(2,20,2):
        getEntropyAnalysis('paciente1', ['pre', 'post'], datosP1,'sampen', sesiones1, metric, 1, 240, 1, 0.8)
for metric in np.arange(3,20,2):
        getEntropyAnalysis('paciente1', ['pre', 'post'], datosP1,'permen', sesiones1, metric, 1, 240, 1, 0.8)
        
# now for the multiescale entropy
for scale in np.arange(2,20,2):
        getEntropyAnalysis('paciente1', ['pre', 'post'], datosP1,'msen', sesiones1, 2, scale, 240, 1, 0.8)
        getEntropyAnalysis('paciente1', ['pre', 'post'], datosP1,'msen', sesiones1, 3, scale, 240, 1, 0.8)

"""
datosP2 = []
sesiones2 = np.array([1,2,3,4])

for ses in sesiones2:
    datos = {}
    # presesion
    try:
        datos['pre'] = pd.read_csv('../data/P2/P2_pre_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['pre'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    try:
        datos['post'] = pd.read_csv('../data/P2/P2_post_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['post'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    datosP2.append(datos)


# for sample and permutation entropies
for metric in np.arange(2,20,2):
        getEntropyAnalysis('paciente2', ['pre', 'post'], datosP2,'sampen', sesiones2, metric, 1, 240, 1, 0.8)
for metric in np.arange(3,20,2):
        getEntropyAnalysis('paciente2', ['pre', 'post'], datosP2,'permen', sesiones2, metric, 1, 240, 1, 0.8)
        
# now for the multiescale entropy
for scale in np.arange(2,20,2):
        getEntropyAnalysis('paciente2', ['pre', 'post'], datosP2,'msen', sesiones2, 2, scale, 240, 1, 0.8)
        getEntropyAnalysis('paciente2', ['pre', 'post'], datosP2,'msen', sesiones2, 3, scale, 240, 1, 0.8)
"""
datosP3 = []
sesiones3 = np.array([1,2,3,4,5])

for ses in sesiones3:
    datos = {}
    # presesion
    try:
        datos['pre'] = pd.read_csv('../data/P3/P3_pre_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['pre'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    try:
        datos['post'] = pd.read_csv('../data/P3/P3_post_sesion_%d.txt'%ses, index_col=False, header=6)
    except:
        datos['post'] = pd.DataFrame(columns=['EEG1','EEG2','EMG','ECG'])
        
    datosP3.append(datos)


# for sample and permutation entropies
for metric in np.arange(2,20,2):
        getEntropyAnalysis('paciente3', ['pre', 'post'], datosP3,'sampen', sesiones3, metric, 1, 240, 1, 0.8)
for metric in np.arange(3,20,2):
        getEntropyAnalysis('paciente3', ['pre', 'post'], datosP3,'permen', sesiones3, metric, 1, 240, 1, 0.8)
        
# now for the multiescale entropy
for scale in np.arange(2,20,2):
        getEntropyAnalysis('paciente3', ['pre', 'post'], datosP3,'msen', sesiones3, 2, scale, 240, 1, 0.8)
        getEntropyAnalysis('paciente3', ['pre', 'post'], datosP3,'msen', sesiones3, 3, scale, 240, 1, 0.8)
"""