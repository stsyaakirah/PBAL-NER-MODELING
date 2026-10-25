# PBAL NER Modeling — WNUT17 Emerging Entities

Repository ini berisi **dataset, source code, hasil eksperimen, dan laporan** implementasi Named Entity Recognition (NER) pada dataset **WNUT17 Emerging and Rare Entities**.

> **Implementasi akhir:** Bi-LSTM–CRF dengan fitur *frozen* RoBERTa di [`wnut17_stage2/opsi1_roberta.py`](wnut17_stage2/opsi1_roberta.py).

## Ringkasan

Proyek membandingkan tiga pendekatan: Conditional Random Field (CRF), Linear SVM, dan Bi-LSTM–CRF. Eksperimen terakhir menambahkan representasi kontekstual RoBERTa-base yang dibekukan ke Bi-LSTM–CRF dan menghasilkan performa terbaik pada split resmi WNUT17.

Dataset menggunakan skema BIO/IOB2 dengan enam tipe entitas: `person`, `location`, `corporation`, `product`, `creative-work`, dan `group`.

## File Utama untuk Pengumpulan

| Kebutuhan | Lokasi |
|---|---|
| Dataset train | [`wnut17_stage2/data/wnut17train.conll`](wnut17_stage2/data/wnut17train.conll) |
| Dataset development | [`wnut17_stage2/data/emerging.dev.conll`](wnut17_stage2/data/emerging.dev.conll) |
| Dataset test berlabel | [`wnut17_stage2/data/emerging.test.annotated`](wnut17_stage2/data/emerging.test.annotated) |
| Implementasi NER akhir | [`wnut17_stage2/opsi1_roberta.py`](wnut17_stage2/opsi1_roberta.py) |
| Ekstraksi RoBERTa | [`wnut17_stage2/extract_roberta.py`](wnut17_stage2/extract_roberta.py) |
| Arsitektur Bi-LSTM–CRF | [`wnut17_stage2/bilstm_crf.py`](wnut17_stage2/bilstm_crf.py) |
| Implementasi CRF dan SVM | [`wnut17_stage2/trainonly_models.py`](wnut17_stage2/trainonly_models.py) |
| Pemrosesan data | [`wnut17_stage2/wnut_data.py`](wnut17_stage2/wnut_data.py) |
| Fitur dan evaluasi | [`features.py`](wnut17_stage2/features.py), [`evalx.py`](wnut17_stage2/evalx.py) |
| Hasil model akhir | [`results/bilstm_roberta.json`](wnut17_stage2/results/bilstm_roberta.json) |
| Prediksi akhir | [`results/pred_to_bilstm_roberta_test.json`](wnut17_stage2/results/pred_to_bilstm_roberta_test.json) |
| Notebook | [`Stage2_NER_WNUT17_Kelompok1_TwitterOnly.ipynb`](wnut17_stage2/Stage2_NER_WNUT17_Kelompok1_TwitterOnly.ipynb) |
| Laporan akhir | [PDF laporan](wnut17_stage2/NER_WNUT17_Comprehensive_Report_TwitterOnly_Seed42.pdf) |

## Struktur Repository

```text
PBAL-NER-MODELING/
├── source/                         # Data sumber/domain tambahan
├── submissions/                    # Prediksi sistem peserta WNUT17
├── stage2/                         # Implementasi awal/ringkas
└── wnut17_stage2/                  # Pipeline eksperimen final
    ├── data/                       # Dataset split resmi
    ├── data_partB/                 # Split gabungan untuk ablasi
    ├── results/                    # Metrik dan prediksi utama
    ├── results_partB/              # Hasil Section B
    ├── trainonly_models.py         # CRF dan SVM final
    ├── bilstm_crf.py               # Arsitektur Bi-LSTM–CRF
    ├── extract_roberta.py          # Ekstraksi fitur RoBERTa
    └── opsi1_roberta.py            # Implementasi NER terbaik
```

Folder `stage2/` adalah implementasi awal. Untuk membaca, menjalankan, atau menilai implementasi terakhir, gunakan **`wnut17_stage2/`**.

## Dataset dan Protokol

| Split | Domain | File |
|---|---|---|
| Train | Twitter | `wnut17train.conll` |
| Development | YouTube | `emerging.dev.conll` |
| Test | Reddit, StackExchange, Twitter | `emerging.test.annotated` |

Model hanya dilatih menggunakan train. Development set digunakan untuk memilih konfigurasi, early stopping, dan decoding bias. Test set hanya digunakan untuk evaluasi akhir. Pipeline memakai `emerging.test.annotated` sebagai *gold test*.

Dataset WNUT17 yang disertakan mengikuti lisensi **CC BY 4.0**, sesuai dokumentasi proyek.

### Section B

`data_partB/` adalah ablasi dengan seluruh data digabung lalu dibagi 80/10/10 berdasarkan dokumen. Hasil Section B **tidak dapat dibandingkan langsung** dengan shared task karena pencampuran meningkatkan kemunculan ulang entitas antara train dan test.

## Model

### CRF

Menggunakan fitur token, konteks, POS, kapitalisasi, dan bentuk kata. Konfigurasi final: L-BFGS, `c1=0.3`, `c2=0.003`, maksimum 300 iterasi, serta seluruh kemungkinan transisi aktif.

### Linear SVM

Bekerja pada level token dengan fitur sparse dan GloVe. Konfigurasi final: `C=0.03`, balanced class weights, dan seed 42. BIO repair tidak dipakai karena menurunkan F1 development.

### Bi-LSTM–CRF

Input baseline menggabungkan frozen GloVe 200d, character CNN, dan capitalization embedding. Representasi diproses oleh Bi-LSTM 128 unit per arah, lalu CRF dan Viterbi decoding.

### Bi-LSTM–CRF + Frozen RoBERTa

Model terbaik menambahkan vektor RoBERTa-base 768d pada setiap token:

```text
GloVe + Character CNN + Capitalization + RoBERTa
                         ↓
                    Bi-LSTM
                         ↓
                        CRF
                         ↓
                 Viterbi decoding
```

RoBERTa hanya menjadi ekstraktor fitur dan **tidak di-fine-tune**. Konfigurasi utama: seed 42, hidden 128 per arah, dropout 0.5, learning rate `1e-3`, batch size 32, Adam, dan gradient clipping 5.0.

## Hasil Implementasi Akhir

| Model | Precision | Recall | Entity F1 | Surface F1 |
|---|---:|---:|---:|---:|
| CRF | 31.95 | 11.40 | 16.80 | 17.61 |
| Linear SVM | 31.51 | 28.27 | 29.80 | 26.02 |
| Bi-LSTM–CRF baseline | 57.22 | 30.86 | 40.10 | 39.03 |
| Bi-LSTM ensemble | — | — | 41.35 | 39.63 |
| Bi-LSTM–CRF + RoBERTa, delta 0 | 64.71 | 44.02 | 52.40 | 51.65 |
| **Bi-LSTM–CRF + RoBERTa, delta 1.0** | **60.81** | **48.75** | **54.12** | **52.94** |

`delta=1.0` dipilih melalui development set. Nilai ini mengurangi bias decoder terhadap label `O`, sehingga recall meningkat.

Hasil lengkap terdapat di `results/to_crf.json`, `results/to_svm.json`, `results/improve_bilstm.json`, dan `results/bilstm_roberta.json`.

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r wnut17_stage2/requirements.txt
pip install transformers torch
```

Pada Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r wnut17_stage2/requirements.txt
pip install transformers torch
```

`transformers` dan `torch` diperlukan untuk ekstraksi RoBERTa, tetapi belum tercantum di `requirements.txt`.

## Menjalankan Eksperimen

```bash
cd wnut17_stage2
```

CRF dan SVM:

```bash
python -W ignore trainonly_models.py
```

Bi-LSTM–CRF baseline dan varian (memerlukan GloVe Twitter 200d):

```bash
GLOVE_GZ=/path/to/glove-twitter-200.gz python prep_glove.py
python -W ignore improve_bilstm.py
python -W ignore ensemble_bilstm.py
```

Model final:

```bash
python -W ignore extract_roberta.py
python -W ignore opsi1_roberta.py
```

Output utama: `results/bilstm_roberta.json` dan `results/pred_to_bilstm_roberta_test.json`.

## Artefak Besar

File `glove-twitter-200.gz`, `glove_sub.npz`, `results/roberta_*.pkl`, dan `results/ckpt_*.pkl` diabaikan oleh `.gitignore`. File tersebut dapat dibuat ulang menggunakan `prep_glove.py`, `extract_roberta.py`, dan script training terkait. Karena checkpoint tidak disertakan, reproduksi penuh memerlukan training ulang.

## Metrik dan Reproducibility

- **Entity F1:** benar jika batas dan tipe entitas tepat.
- **Surface F1:** mengukur pasangan unik antara surface text dan tipe entitas.
- Eksperimen final menggunakan seed 42.
- Model dan hyperparameter dipilih melalui development set.
- Test set tidak dipakai untuk memperbarui bobot.
- Frozen RoBERTa adalah peningkatan modern, bukan perbandingan setara dengan sistem shared task 2017.
- Section B adalah studi ablasi, bukan hasil utama.

## Referensi

- Derczynski, L., Nichols, E., van Erp, M., & Limsopatham, N. (2017). *Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition*.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.

## Team Members

- Amelia Gizzela Sheehan Auni (25/574579/PPA/07252)
- Aulia Gita Pratiwi (25/572701/PPA/07197)
- St. Syakirah (25/573284/PPA/07217)
- Aditya Iko Winarko (25/574566/PPA/07251)
- Royan Aditya (26/589551/PPA/07364)
