import os
import csv
import numpy as np
from tqdm import tqdm
from .preprocess import preprocessWav
from .stft import computeSpectogram
from .svd import (
    computeSVD,
    dominantRatio,
    cumulativeEnergy,
    frobeniusErrorFromSingular,
    singularSpread,
)


FEATURE_KEYS = [
    'dominant_ratio',
    'cumulative_energy_1',
    'cumulative_energy_3',
    'singular_spread',
    'frobenius_error_1',
    'frobenius_error_3',
]


def _compute_feature_row(seg, sr, nFFT, hop, toDB, label=None, fname=None):
    S = computeSpectogram(seg, sr, nFFT, hop, toDB=toDB)
    _, s, _ = computeSVD(S, full=False)
    row = {
        'file': fname if fname else '',
        'label': label if label else '',
        'dominant_ratio': float(dominantRatio(s)),
        'cumulative_energy_1': float(cumulativeEnergy(s, threshold=1)),
        'cumulative_energy_3': float(cumulativeEnergy(s, threshold=3 if len(s) >= 3 else len(s))),
        'singular_spread': float(singularSpread(s)),
        'frobenius_error_1': float(frobeniusErrorFromSingular(s, 1)),
        'frobenius_error_3': float(frobeniusErrorFromSingular(s, 3)),
        'num_svals': int(len(s)),
    }
    return row

def analyzeDir(dir, output, label, sr=44100, nFFT=1024, hop=256, toDB=False):
    rows = []
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.wav')])
    for fpath in tqdm(files, desc=f'Analyzing {label}'):
        seg, sr = preprocessWav(fpath, sr)
        row = _compute_feature_row(seg, sr, nFFT, hop, toDB, label=label, fname=os.path.basename(fpath))
        rows.append(row)
    _write_rows(output, rows)
    return rows


def _write_rows(output, rows):
    if not rows:
        return
    writeHeader = not os.path.exists(output)
    with open(output, 'a', newline='') as csvfile:
        fieldnames = ['file', 'label', 'dominant_ratio', 'cumulative_energy_1', 'cumulative_energy_3', 'singular_spread', 'frobenius_error_1', 'frobenius_error_3', 'num_svals']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if writeHeader:
            writer.writeheader()
        writer.writerows(rows)
    
def run(dataRoot ='data', outputCSV='results/metrics.csv'):
    os.makedirs(os.path.dirname(outputCSV), exist_ok=True)
    if os.path.exists(outputCSV):
        os.remove(outputCSV)
    perfect = analyzeDir(os.path.join(dataRoot,'perfect_hit'), outputCSV, label='perfect')
    off = analyzeDir(os.path.join(dataRoot,'offcenter_hit'), outputCSV, label='offcenter')
    return perfect + off


def fit_centroid_classifier(rows):
    if not rows:
        raise ValueError('No rows provided to train classifier')
    data = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float64)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-12
    zdata = (data - mean) / std
    labels = np.array([r['label'] for r in rows])
    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroids[label] = zdata[mask].mean(axis=0)
    return {'mean': mean, 'std': std, 'centroids': centroids}


def classify_features(row, stats):
    feats = np.array([row[k] for k in FEATURE_KEYS], dtype=np.float64)
    z = (feats - stats['mean']) / stats['std']
    best_label = None
    best_dist = None
    for label, center in stats['centroids'].items():
        dist = np.linalg.norm(z - center)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label, best_dist


def classifyDir(dir, stats, predOutput='results/predictions.csv', sr=44100, nFFT=1024, hop=256, toDB=False):
    if os.path.exists(predOutput):
        os.remove(predOutput)
    rows = []
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.wav')])
    for fpath in tqdm(files, desc='Classifying unknown'):
        seg, sr = preprocessWav(fpath, sr)
        row = _compute_feature_row(seg, sr, nFFT, hop, toDB, label='', fname=os.path.basename(fpath))
        pred, dist = classify_features(row, stats)
        row['predicted_label'] = pred
        row['distance'] = float(dist)
        rows.append(row)
    writeHeader = not os.path.exists(predOutput)
    fieldnames = ['file', 'label', 'predicted_label', 'distance'] + FEATURE_KEYS + ['num_svals']
    os.makedirs(os.path.dirname(predOutput), exist_ok=True)
    with open(predOutput, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if writeHeader:
            writer.writeheader()
        if rows:
            writer.writerows(rows)
    return rows


def run_with_classifier(dataRoot='data', outputCSV='results/metrics.csv', classifyDirPath=None, predOutput='results/predictions.csv'):
    rows = run(dataRoot=dataRoot, outputCSV=outputCSV)
    stats = fit_centroid_classifier(rows)
    if classifyDirPath:
        classifyDir(classifyDirPath, stats, predOutput=predOutput)
    return rows, stats
