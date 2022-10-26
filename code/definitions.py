import numpy as np
import pandas as pd
import antropy as ent
from scipy import signal
from scipy import stats
from scipy.fft import fft, fftfreq 
from os.path import exists
import matplotlib.pyplot as plt

# The EEG threshold is defined here
eeg_thres = 200

# La transformada de Fourier
def getFourierTransform(data, measfreq, maxfreq):
    """ This function calculates the fourier transform of a timeseries in the logarithmic scale, the frequency boundaries must be provided """
    data_fft = 10*np.log10(np.abs(fft(data))**2)
    freqs = fftfreq(len(data_fft), 1/measfreq)
    mask = (freqs > 0) & (freqs < maxfreq)
    return freqs[mask],data_fft[mask]

# Permutation entropy
def get_perm_entropy(data, metric, measfreq, dtime, overlap):
    """ This function calculates the permutation entropy for a time series, the data are divided in windows of width dtime/(1-overlap), the entropy is only calculated if the interval does not exceeds the defined threshold"""
    timewindow = dtime/(1-overlap)
    ndata = int(timewindow*measfreq)
    step = int(measfreq*dtime)
    entropy = []
    times = []
    for i in np.arange(np.floor((len(data)-ndata)/step).astype(int)+1):
        newdata = data[i*step:i*step+ndata]
        if (np.max(np.abs(newdata))<eeg_thres):
            times.append((i*step+ndata/2)/measfreq)
            entropy.append(ent.perm_entropy(data[i*step:i*step+ndata], order=metric, normalize = True))
    return np.array(times), np.array(entropy)

# Sample entropy
def get_samp_entropy(data, metric, measfreq, dtime, overlap):
    """ This function calculates the sample entropy for a time series, the data are divided in windows of width dtime/(1-overlap), the entropy is only calculated if the interval does not exceeds the defined threshold"""
    timewindow = dtime/(1-overlap)
    ndata = int(timewindow*measfreq)
    step = int(measfreq*dtime)
    entropy = []
    times = []
    for i in np.arange(np.floor((len(data)-ndata)/step).astype(int)+1):
        newdata = data[i*step:i*step+ndata]
        if (np.max(np.abs(newdata))<eeg_thres):
            times.append((i*step+ndata/2)/measfreq)
            entropy.append(ent.sample_entropy(data[i*step:i*step+ndata], order=metric))
    return np.array(times), np.array(entropy)

# MultiScale Entropy
def get_ms_entropy(data, metric, scale, measfreq, dtime, overlap):
    """ This function calculates the sample entropy for a time series, the data are divided in windows of width scale*dtime/(1-overlap), the entropy is only calculated if the interval does not exceeds the defined threshold"""
    timewindow = scale*dtime/(1-overlap)  # the length of the time window
    ndata = int(timewindow*measfreq)  # the number of data in each time window
    step = int(measfreq*dtime*scale)  # the number of data in each step
    entropy = []
    times = []
    nndata = np.floor(ndata/scale).astype(int)
    for i in np.arange(np.floor((len(data)-ndata)/step).astype(int)+1):
        newdata = data[i*step:i*step+ndata] # the new data
        if (np.max(np.abs(newdata))<eeg_thres):
            msdata = []
            for j in np.arange(nndata):
                msdata.append(newdata[j*scale:(j+1)*scale].mean()) # the average of the data
            times.append((i*step+ndata/2)/measfreq) # the time average of the interval
            entropy.append(ent.sample_entropy(np.array(msdata), order=metric)) # The calculation of the multiscale entropy
    return np.array(times), np.array(entropy)

# Figura espectral por bandas
def createSpectralFigureBands(subject, data, Nsesion, sesiones, kind, eeg, metric, measfreq, maxfreq, dtime, overlap):
    """This function creaste a spectral figure where the entropy data is shown for each of the bands"""
    # primero calculamos la FFT
    freqs_fft, data_fft = getFourierTransform(data[Nsesion][kind][eeg].values, measfreq, maxfreq)
    # definimos las bandas
    bands = [4,8,12,35]
    bandnames = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    perments = []
    sampents = []
    
    for i in range(len(bandnames)):
        if i == 0: # first band
            sos = signal.butter(8,bands[i], btype = 'lowpass', output = 'sos', fs = 240)
            filtered = signal.sosfilt(sos, data[Nsesion][kind][eeg])
            perments.append(get_perm_entropy(filtered, metric, measfreq, dtime, overlap))
            sampents.append(get_samp_entropy(filtered, metric, measfreq, dtime, overlap))
        elif i == (len(bandnames)-1): # last band
            sos = signal.butter(8,bands[i-1], btype = 'highpass', output = 'sos', fs = 240)
            filtered = signal.sosfilt(sos, data[Nsesion][kind][eeg])
            perments.append(get_perm_entropy(filtered, metric, measfreq, dtime, overlap))
            sampents.append(get_samp_entropy(filtered, metric, measfreq, dtime, overlap))
        else: # middle bands
            sos = signal.butter(8,[bands[i-1],bands[i]], btype = 'bandpass', output = 'sos', fs = 240)
            filtered = signal.sosfilt(sos, data[Nsesion][kind][eeg])
            perments.append(get_perm_entropy(filtered, metric, measfreq, dtime, overlap))
            sampents.append(get_samp_entropy(filtered, metric, measfreq, dtime, overlap))
            
    
    # The figure parameters
    # individual width
    wi=3.5
    # individual height
    hi=2.0
    # left margin
    lm=0.72
    # right margin
    rm=0.1
    # bottom margin
    bm=0.45
    # top margin
    tm=0.7
    # horizontal gap
    hg=0.3
    # vertical gap
    vg=0.5
    # number of columns
    nc=1
    # number of rows
    nr=5

    # The calculations
    sx=wi*nc+lm+rm+hg*(nc-1)  # Figure x
    sy=hi*nr+bm+tm+vg*(nr-1)  # Figure y
    t=(sy-tm)/sy
    b=bm/sy
    l=lm/sx
    r=(sx-rm)/sx
    hr=vg/hi
    wr=hg/wi
    
    # ahora creamos la gráfica de los tres elementos
    fig, ax = plt.subplots(nr,nc,figsize=(sx,sy))
    fig.suptitle('Estudio espectral:\n '+subject+' '+str(sesiones[Nsesion])+', '+kind+', señal '+eeg, fontsize=16)
    # primero ponemos los datos originales
    ax[0].plot(np.arange(len(data[Nsesion][kind][eeg]))/measfreq,data[Nsesion][kind][eeg])
    ax[0].set_xlabel('tiempo (s)')
    ax[0].set_ylabel('señal EEG')
    # posteriormente ponemos el spectrograma
    #ax[2].imshow(np.transpose(spec)[::-1,:], aspect='auto', extent = [times[0],times[-1],freqs_spec[0],freqs_spec[-1]])
    window = dtime/(1-overlap)
    aux,aux,aux,spc=ax[3].specgram(data[Nsesion][kind][eeg], Fs=measfreq, scale = 'dB', NFFT = int(measfreq*window), noverlap = int((window-dtime)*measfreq), mode = 'magnitude', cmap = 'jet')
    ax[0].set_xlim(ax[3].get_xlim())
    ax[3].set_xlabel('tiempo (s)')
    ax[3].set_ylabel('frecuencia (Hz)')
    # finalmente, la transformada de fourier completa
    ax[4].plot(freqs_fft,data_fft)
    ax[4].set_xlabel('frecuencia (Hz)')
    ax[4].set_ylabel('potencia (dB)')
    ax[4].set_xlim(0,maxfreq)
    # Ahora la entropía de permutación
    for i in range(len(bandnames)):
        ax[1].plot(perments[i][0],perments[i][1], label= 'perm_ent_'+bandnames[i], color='C%d'%i)
        ax[1].axhline(y=perments[i][1].mean(), color='C%d'%i, linestyle='--')
    ax[1].set_xlabel('tiempo (s)')
    ax[1].set_ylabel('entropía')
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(ax[3].get_xlim())
    ax[1].legend(fontsize = 6)
    # ahora la entropía de la muestra
    for i in range(len(bandnames)):
        ax[2].plot(sampents[i][0],sampents[i][1], label= 'samp_ent_'+bandnames[i], color='C%d'%i)
        ax[2].axhline(y=sampents[i][1].mean(), color='C%d'%i, linestyle='--')
    ax[2].set_xlabel('tiempo (s)')
    ax[2].set_ylabel('entropía')
    ax[2].set_ylim(0,1)
    ax[2].set_xlim(ax[3].get_xlim())
    ax[2].legend(fontsize = 6)
    # The adjustment
    plt.subplots_adjust(wspace=wr,hspace=hr,bottom=b, top=t, left=l, right=r)
    plt.savefig('images/sp_'+subject+'_%d_'%(sesiones[Nsesion])+kind+'_'+eeg+'.png', dpi = 200)
    plt.close()


# Cálculo general de entropía
def getEntropyAnalysis(subject, phases, data, ent_kind, sesiones, metric, scale, measfreq, dtime, overlap):
    """This function calculates the entropy for all the sessions of the given subject and creates all the comparisons between sessions

    Args:
        subject (string): The subject, ex: P1, P3
        phases (string list]): The measurement phases
        data (dataframe): The dataframe with all the data
        ent_kind (_type_): The kind of entropy to calculate
        sesiones (_type_): The sessions
        metric (_type_): the metric
        scale (_type_): the scale
        measfreq (_type_): the measurement frequency
        dtime (_type_): the step between dta slices
        overlap (_type_): the overlap between data slices
    """
    # data, the list with all the data
    # ent_kind, a string: either 'sampen', 'permen' or 'msen'
    if ent_kind not in ['sampen', 'permen', 'msen']:
        print("The kind of entropy is not correct")
        return 
    # metric, the metric to calculate the entropy
    # scale, needed for 'msen' the scale for multiscale entropy, not used for sampen or permen
    # measfreq, the frequency of the measurement
    # dtime, the interval step
    # overlap, the interval overlap. The interval size is calculated as dtime/(1-overlap)
    
    # we first check whether the entropy has already been calculated
    if ent_kind == 'msen':
        fileroot = 'results/entropy_'+subject+'_'+ent_kind+'_'+str(metric)+'_'+str(scale)+'_'+str(dtime)+'_'+str(overlap)
    else:
        fileroot = 'results/entropy_'+subject+'_'+ent_kind+'_'+str(metric)+'_'+str(dtime)+'_'+str(overlap)
    
    comparison = phases[0]+phases[1]
    if exists(fileroot+'_'+comparison+'_stats.xlsx'):  # the file exists and we can proceed
        return
    else:
        # we define a dataframe with the entropy calculations
        ent_data_band = pd.DataFrame()

        # now we define lists to store the information
        ents = []
        eegs = []
        bands = []
        sesions = []
        timing = []
        bands = [4,8,12,35]
        bandnames = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all bands']
        banddata = []

        for i,ses  in enumerate(sesiones):
            for key in phases:
                for eeg in ['EEG1','EEG2']:
                    for k in range(len(bandnames)):
                        try:
                            if k == 0: # first band
                                sos = signal.butter(8, bands[k], btype = 'lowpass', output = 'sos', fs = measfreq)
                                filtered = signal.sosfilt(sos, data[i][key][eeg])
                                if ent_kind == 'sampen':
                                    times, ent = get_samp_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'permen':
                                    times, ent = get_perm_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'msen':
                                    times, ent = get_ms_entropy(filtered, metric, scale, measfreq, dtime, overlap)

                            elif k == len(bandnames)-1: # all bands
                                if ent_kind == 'sampen':
                                    times, ent = get_samp_entropy(data[i][key][eeg], metric, measfreq, dtime, overlap)
                                elif ent_kind == 'permen':
                                    times, ent = get_perm_entropy(data[i][key][eeg], metric, measfreq, dtime, overlap)
                                elif ent_kind == 'msen':
                                    times, ent = get_ms_entropy(data[i][key][eeg], metric, scale, measfreq, dtime, overlap)

                            elif k == (len(bandnames)-2): # last band
                                sos = signal.butter(8,bands[k-1], btype = 'highpass', output = 'sos', fs = measfreq)
                                filtered = signal.sosfilt(sos, data[i][key][eeg])
                                if ent_kind == 'sampen':
                                    times, ent = get_samp_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'permen':
                                    times, ent = get_perm_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'msen':
                                    times, ent = get_ms_entropy(filtered, metric, scale, measfreq, dtime, overlap)

                            else: # middle bands
                                sos = signal.butter(8,[bands[k-1],bands[k]], btype = 'bandpass', output = 'sos', fs = measfreq)
                                filtered = signal.sosfilt(sos, data[i][key][eeg])
                                if ent_kind == 'sampen':
                                    times, ent = get_samp_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'permen':
                                    times, ent = get_perm_entropy(filtered, metric, measfreq, dtime, overlap)
                                elif ent_kind == 'msen':
                                    times, ent = get_ms_entropy(filtered, metric, scale, measfreq, dtime, overlap)

                            banddata = banddata + [bandnames[k]]*len(ent)
                            ents = ents +list(ent)
                            eegs = eegs +[eeg]*len(ent)
                            sesions = sesions + [ses]*len(ent)
                            timing = timing + [key]*len(ent)
                        except:
                            pass


        ent_data_band['sesion']=sesions
        ent_data_band['entropia'] = ents
        ent_data_band['eeg'] = eegs
        ent_data_band['banda'] = banddata
        ent_data_band['fase'] = timing

        # calculating means and stds
        ent_mean = ent_data_band.groupby(['sesion', 'eeg', 'banda', 'fase']).mean().reset_index().rename({'entropia':'mean'}, axis = 1)
        ent_std = ent_data_band.groupby(['sesion', 'eeg', 'banda', 'fase']).std().reset_index().rename({'entropia':'std'}, axis = 1)

        # Evaluating the statistical difference among the 'pre' measurements among sessions
        statsPRE = pd.DataFrame() 

        pvals = []
        diffs = []
        stds = []
        eegs = []
        bands = []
        sess = []


        for i,ses  in enumerate(sesiones[1:]):
            for j,eeg in enumerate(['EEG1','EEG2']):
                for k,name in enumerate(bandnames):
                    bands.append(name)
                    eegs.append(eeg)
                    sess.append(ses)
                    mask1 = (ent_data_band['sesion']==sesiones[i]) & (ent_data_band['eeg']==eeg) & (ent_data_band['banda']==name) & (ent_data_band['fase']==phases[0])
                    mask2 = (ent_data_band['sesion']==ses) & (ent_data_band['eeg']==eeg) & (ent_data_band['banda']==name) & (ent_data_band['fase']==phases[0])
                    ttestres = stats.ttest_ind(ent_data_band.loc[mask1,'entropia'],ent_data_band.loc[mask2,'entropia'])    
                    pvals.append(ttestres.pvalue)
                    mask3 = (ent_mean['sesion']==sesiones[i]) & (ent_mean['eeg']==eeg) & (ent_mean['banda']==name) & (ent_mean['fase']==phases[0])
                    mask4 = (ent_mean['sesion']==ses) & (ent_mean['eeg']==eeg) & (ent_mean['banda']==name) & (ent_mean['fase']==phases[0])
                    try:
                        diffs.append((100*(ent_mean.loc[mask4,'mean'].values-ent_mean.loc[mask3,'mean'].values)/ent_mean.loc[mask3,'mean'].values)[0])
                        stds.append((100*np.sqrt(ent_std.loc[mask4,'std'].values**2+ent_std.loc[mask3,'std'].values)/ent_mean.loc[mask3,'mean'].values)[0])
                    except:
                        diffs.append(np.nan)
                        stds.append(np.nan)
                        

        statsPRE['sesion'] = sess
        statsPRE['eeg'] = eegs
        statsPRE['bands'] = bands
        statsPRE['pvalues'] = pvals
        statsPRE['diff'] = diffs
        statsPRE['errdiff'] = stds


        # Evaluating the statistical difference among the 'pre' and 'post' measurements in each session
        statsPREPOST = pd.DataFrame() 

        pvals = []
        diffs = []
        stds = []
        eegs = []
        bands = []
        sess = []

        for i,ses  in enumerate(sesiones):
            for j,eeg in enumerate(['EEG1','EEG2']):
                for k,name in enumerate(bandnames):
                    bands.append(name)
                    eegs.append(eeg)
                    sess.append(ses)
                    mask1 = (ent_data_band['sesion']==ses) & (ent_data_band['eeg']==eeg) & (ent_data_band['banda']==name) & (ent_data_band['fase']==phases[0])
                    mask2 = (ent_data_band['sesion']==ses) & (ent_data_band['eeg']==eeg) & (ent_data_band['banda']==name) & (ent_data_band['fase']==phases[1])
                    ttestres = stats.ttest_ind(ent_data_band.loc[mask1,'entropia'],ent_data_band.loc[mask2,'entropia'])    
                    pvals.append(ttestres.pvalue)
                    mask3 = (ent_mean['sesion']==ses) & (ent_mean['eeg']==eeg) & (ent_mean['banda']==name) & (ent_mean['fase']==phases[0])
                    mask4 = (ent_mean['sesion']==ses) & (ent_mean['eeg']==eeg) & (ent_mean['banda']==name) & (ent_mean['fase']==phases[1])
                    try:
                        diffs.append((100*(ent_mean.loc[mask4,'mean'].values-ent_mean.loc[mask3,'mean'].values)/ent_mean.loc[mask3,'mean'].values)[0])
                        stds.append((100*np.sqrt(ent_std.loc[mask4,'std'].values**2+ent_std.loc[mask3,'std'].values)/ent_mean.loc[mask3,'mean'].values)[0])
                    except:
                        diffs.append(np.nan)
                        stds.append(np.nan)

        statsPREPOST['sesion'] = sess
        statsPREPOST['eeg'] = eegs
        statsPREPOST['bands'] = bands
        statsPREPOST['pvalues'] = pvals
        statsPREPOST['diff'] = diffs
        statsPREPOST['errdiff'] = stds

        statsPRE['signif'] = statsPRE['pvalues'] < 0.05
        statsPREPOST['signif'] = statsPREPOST['pvalues'] < 0.05



        #plt.close()

        #exporting the results as excel files
        ent_data_band.to_excel(fileroot+'_ent.xlsx')
        statsPRE.to_excel(fileroot+'_'+phases[0]+'_stats.xlsx')
        statsPREPOST.to_excel(fileroot+'_'+comparison+'_stats.xlsx')

    return 

