# Stage 2: NER WNUT17 dengan CRF, SVM, dan Bi-LSTM CRF

Implementasi tiga model NER untuk dataset WNUT17 Emerging and Rare Entities,
sesuai Assignment #2. Semua model memakai modul data, fitur, dan evaluasi yang sama
agar perbandingannya adil.

## Struktur

```
stage2/
  ner/
    data.py        # baca CoNLL, normalisasi token, word shape
    features.py    # fitur per token (konteks +/-2, affix, shape, POS) untuk CRF & SVM
    evaluate.py    # F1 entity (seqeval strict) + F1 surface + confusion matrix + contoh
  train_crf.py         # model CRF (sklearn-crfsuite)
  train_svm.py         # model SVM (LinearSVC + DictVectorizer, BIO repair)
  train_bilstm_crf.py  # model Bi-LSTM CRF (PyTorch + torchcrf)
  validate_scorer.py   # bukti modul evaluasi benar (reproduksi Tabel 2 paper)
  outputs/             # hasil *.json dan log
  requirements.txt
```

Data (`wnut17train.conll`, `emerging.dev.conll`, `emerging.test.annotated`) diambil
dari folder Assignment 2 di atas `stage2/`. Path diatur di `ner/data.py`.

## Menjalankan

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

python validate_scorer.py         # cek modul evaluasi (harus cocok dengan paper)
python train_crf.py               # CRF
python train_svm.py               # SVM
python train_bilstm_crf.py --epochs 25   # Bi-LSTM CRF (CPU ~10 dtk/epoch)
```

Semua skrip memakai seed 42, mencetak F1 entity + F1 surface untuk dev dan test,
laporan per tipe, confusion matrix, lalu menyimpan hasil ke `outputs/*.json`.

## Evaluasi

Dua metrik resmi WNUT17, keduanya level entity:
- **F1 entity**: exact match batas dan tipe (`seqeval`, mode strict IOB2).
- **F1 surface**: pasangan unik (surface form, tipe); entitas berulang dihitung sekali.
  Rilis data tidak menyertakan scorer resmi, jadi metrik ini diimplementasikan sendiri
  di `ner/evaluate.py` dan **divalidasi** dengan `validate_scorer.py`: menilai 7
  submission peserta dan mencocokkan hasilnya dengan Tabel 2 paper (mis. UH-RiTUAL
  entity 41,86 dan surface 40,24 tereproduksi persis).

## Catatan hasil

WNUT17 memang sulit: 96-98% entitas dev/test tidak pernah muncul di train, sehingga
fitur identitas kata hampir tidak membantu. Sistem terbaik shared task hanya mencapai
F1 entity 41,86. Baseline fitur-leksikal (CRF/SVM di sini) wajar berada di bawah itu,
terutama pada recall.

Langkah peningkatan yang sudah direncanakan (Stage 1) dan menjadi pekerjaan lanjutan:
- **Fitur embedding / cluster** untuk CRF dan SVM (mis. cluster K-Means atas GloVe
  Twitter) agar dapat menggeneralisasi ke kata tak terlihat -> menaikkan recall.
- **GloVe Twitter 200d** untuk Bi-LSTM CRF lewat `--glove path/ke/glove.twitter.27B.200d.txt`
  (unduh dari https://nlp.stanford.edu/projects/glove/). Embedding acak dari awal
  (default) menghasilkan skor jauh lebih rendah.
- **Tuning hyperparameter** pada dev, lalu latih ulang pada train+dev sebelum
  evaluasi test (praktik umum WNUT17).

## Hyperparameter utama

| Model | Argumen | Default |
|---|---|---|
| CRF | `--c1 --c2 --max-iter` | 0.0, 0.1, 100 |
| SVM | `--C` | 0.1 |
| Bi-LSTM CRF | `--epochs --batch-size --lr --word-dim --hidden --glove` | 20, 16, 1e-3, 100, 200, none |
