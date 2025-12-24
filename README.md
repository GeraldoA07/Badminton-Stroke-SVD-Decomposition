# Badminton Stroke SVD Decomposition
Implementasi SVD untuk klasifikasi pukulan badminton (perfect vs off-center) berbasis fitur spektrum audio.

## Identitas
- Nama: Geraldo Artemius
- NIM: 13524005
- Kelas: K1

## Ringkas Metode
1) Preproses WAV: resample, ekstrak segmen impact, normalisasi puncak.
2) STFT -> matriks spektrum log/amplitude dB.
3) SVD pada matriks spektrum → fitur singular values:
	 - dominant_ratio, cumulative_energy_1/3, singular_spread,
	 - frobenius_error_1/3, num_svals.
4) Klasifikasi centroid sederhana (z-score fitur) antara label latih `perfect` dan `offcenter`.

## Struktur Data
- `data/perfect_hit/*.wav`
- `data/offcenter_hit/*.wav`
- `data/unknown/*.wav`

## Requirements
```
pip install -r requirements.txt
```

## Execution
Jalankan dari root repository:
```
python -m src.run
```
Perilaku:ntoh PowerShell Windows):

## Output
- `results/metrics.csv`: fitur dan label ground-truth untuk data latih.
- `results/predictions.csv`: file, label asli (kosong), predicted_label (perfect atau off-center).
