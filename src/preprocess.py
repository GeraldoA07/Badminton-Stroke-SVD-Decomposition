import numpy as np
import librosa
import soundfile as sf

def resampleWav(file, ori, target_sr=44100):
    if ori == target_sr:
        return file, ori
    ysr = librosa.resample(file.astype(np.float32), orig_sr=ori, target_sr=target_sr)
    return ysr, target_sr  

def rmsEnvelope(y, frameLen=1024, hopLen=256):
    return librosa.feature.rms(y=y, frame_length=frameLen, hop_length=hopLen)[0]

def extractImpactSegment(y, sr, preMS=100, postMS=300, rmsFrame = 1024, hop=256):
    env = rmsEnvelope(y, rmsFrame, hop)
    peakIdx = int(np.argmax(env))
    frameCenter = peakIdx * hop + rmsFrame // 2
    preSample = int(preMS * sr / 1000)
    postSample = int(postMS * sr / 1000)
    start = max(0, frameCenter - preSample)
    end = min(len(y), frameCenter + postSample)
    seg = y[start:end]
    return seg

def normalizePeak(seg):
    maxAmp = np.max(np.abs(seg)) + 1e-12
    return seg/maxAmp

def preprocessWav(path,target_sr=44100, preMS=100, postMS=300):
    y, ori = librosa.load(path, sr=None, mono=True)
    y, sr = resampleWav(y, ori, target_sr)
    seg = extractImpactSegment(y, sr, preMS, postMS)
    seg = normalizePeak(seg)
    return seg, sr
    