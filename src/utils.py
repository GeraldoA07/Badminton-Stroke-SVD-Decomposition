import os
import numpy as np
import soundfile as sf

def listWav(dir):
    files = []
    for ext in (".wav", ".WAV"):
        files.extend(sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext)]))
        
    return files

def readWav(file, target=None):
    y, srNative = sf.read(file)
    if (y.ndim > 1):
        y = np.mean(y, axis=1)
    if target is None or target == srNative:
        return y.astype(np.float32), srNative
    else: 
        return y.astype(np.float32), srNative