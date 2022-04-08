import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks
import math


# ===========================================================
# PEAKS DETECTOR
#
# This function detects the nb_peaks first biggest peaks in a
# vector vec. The magnitude of a peak is defined has the 
# difference between the value at the peak and the value of 
# its neighboor samples.
#
# ARGUMENTS:
#    * vec: 1D-array in which peaks needs to be search.
#    * nb_peaks: the number of peaks to detect.
#
# RETURN:
#    The indices of the nb_peaks found in vec
# ===========================================================
def peaks_detector(vec, nb_peaks=1):
    thrd = 0
    thrd_step = 0.001 * np.max(np.abs(vec))
    peaks_idx, prop = find_peaks(vec, threshold=thrd)
    
    while(len(peaks_idx) > nb_peaks):
        thrd += thrd_step
        peaks_idx, prop = find_peaks(vec, threshold=thrd)
        
    if len(peaks_idx) != nb_peaks:
        bound_a = thrd - thrd_step
        bound_b = thrd
        thrd = (bound_a + bound_b)/2
        peaks_idx, prop = find_peaks(vec, threshold=thrd)
        
        while(len(peaks_idx) != nb_peaks):
            if len(peaks_idx) < nb_peaks:
                bound_b = thrd
            else:
                bound_a = thrd
            
            bound_a = thrd - thrd_step
            bound_b = thrd
            thrd = (bound_a + bound_b)/2
            peaks_idx, prop = find_peaks(vec, threshold=thrd)
        
    return peaks_idx


def plotWave():
    # Extract Raw Audio from Wav File
    sample_rate, data = wavfile.read('./audio/TEDx_echo.wav')
    data = data.astype(float)
    n_data = len(data)
    #Creating time array
    time_data = np.arange(0, n_data)/sample_rate
    #Ploting the signal
    plt.figure(1)
    plt.title("Signal wave (with echo)")
    plt.xlabel("Time [s]")
    plt.plot(time_data, data, color=(0.,0.,0.5))
    plt.show()   


def autocorrelation():
    # Extract Raw Audio from Wav File
    sample_rate, data = wavfile.read('./audio/TEDx_echo.wav')
    data = data.astype(float)

    #Normalisation
    normalized = data/np.mean(data)

    #Watching where does the signal repeats itself with the correlation
    correlation = signal.correlate(normalized, normalized, mode = "full", method='auto')

    #Creating array for y-axis
    l_value = np.linspace((-sample_rate/2+1), (sample_rate/2 -1), len(correlation))

    plt.figure(1)
    plt.title("Autocorrelation between the audio and it's echo")
    plt.xlabel("L value between -N+1 and N-1")
    plt.ylabel("Autocorrelation")
    plt.plot(l_value, correlation)
    plt.show()

def findDelay():
    data = wavfile.read('./audio/TEDx_echo.wav')
    data = data.astype(float)

    #Normalisation
    normalized = data/np.mean(data)

    #Watching where does the signal repeats itself with the correlation
    correlation = signal.correlate(normalized, normalized, mode = "full", method='auto')

    #Geting peak indexes
    ind_peaks = peaks_detector(correlation, 3)

    Delay_1 = ind_peaks[1] - ind_peaks[0]
    Delay_2 = ind_peaks[2] - ind_peaks[1] 

    #Same if we print Delay_1 or Delay_2 cause they are equal
    print('\n Delay = ', Delay_1)


def FindR():
    _,data = wavfile.read('./audio/TEDx_echo.wav')
    #data = data.astype(float)

    #Normalisation
    normalized = data/np.mean(data)

    #Watching where does the signal repeats itself with the correlation
    correlation = signal.correlate(normalized, normalized, mode = "full", method='auto')

    #Geting peak indexes
    ind_peaks = peaks_detector(correlation, 3)

    #Resolvons equation du second degre.
    #R_y_0 = valeur au centre du siganl (vu le graphe) et R_y_D = valeur a la position D plus loin donc au peak suivant
    #Ce qui nous donne le ratio R_y_0/R_y_D
    ratio = correlation[ind_peaks[1]]/correlation[ind_peaks[2]]
    Delta = ratio*ratio - 4
    R = (ratio - math.sqrt(Delta))/2

    print('\nR =  ', R)

#def supressEcho()




if __name__ == "__main__":
       
    #Q1.a
    #plotWave()

    #Q1.b
    #autocorrelation()

    #Q1.d
    #findDelay()
    FindR()

    #Q1.f
