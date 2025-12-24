import os
from .evaluation import run, runWithClassifier

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(project_root, 'data')
    metrics_path = os.path.join(project_root, 'results', 'metrics.csv')
    preds_path = os.path.join(project_root, 'results', 'predictions.csv')

    classify_dir_env = os.getenv('CLASSIFY_DIR')
    classify_dir = None
    if classify_dir_env:
        classify_dir = classify_dir_env if os.path.isabs(classify_dir_env) else os.path.join(project_root, classify_dir_env)
    else:
        default_unknown = os.path.join(data_root, 'unknown')
        if os.path.isdir(default_unknown):
            classify_dir = default_unknown

    if classify_dir and os.path.isdir(classify_dir):
        has_wav = any(f.lower().endswith('.wav') for f in os.listdir(classify_dir))
        if has_wav:
            runWithClassifier(dataRoot=data_root, outputCSV=metrics_path, classifyDirPath=classify_dir, predOutput=preds_path)
            print(f"Selesai! Metrics di {metrics_path} dan prediksi di {preds_path}")
        else:
            run(dataRoot=data_root, outputCSV=metrics_path)
            print(f"Selesai! Metrics di {metrics_path}. Tidak ada WAV di {classify_dir}, jadi prediksi dilewati.")
    else:
        run(dataRoot=data_root, outputCSV=metrics_path)
        if classify_dir:
            print(f"Folder {classify_dir} tidak ditemukan; hanya menulis metrics di {metrics_path}")
        else:
            print("Selesai! Metrics berhasil dibuat di results/metrics.csv")
