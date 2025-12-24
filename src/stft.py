import numpy as np
import librosa

def computeSpectogram(y, sr, nFFT=1024, hopLen=256, window='hann', toDB = False):
    s = np.abs(librosa.stft(y=y, n_fft=nFFT, hop_length=hopLen, window=window))
    if toDB: 
        s = librosa.amplitude_to_db(s, ref=np.max)
    else: 
        s = np.log1p(s)
    return s

def spectogramToMatrix(s): # Sudah frekuensi x waktu jadi langsung return
    return s 