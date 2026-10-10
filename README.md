<div align="center">

# 🧠 Pembelajaran Mendalam (Deep Learning)

**Kumpulan tugas, latihan modul, dan eksperimen mata kuliah Pembelajaran Mendalam (IF25-40401)**

Institut Teknologi Sumatera

*Deep Learning coursework: MLP, transfer learning, regularization, and CNN architectures on CIFAR-10, built with PyTorch.*

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Last commit](https://img.shields.io/github/last-commit/Ahmatsembiring/Pembelajaran-Mendalam)

</div>

---

## Identitas

| | |
|---|---|
| **Nama** | Ahmat Prayoga Sembiring |
| **NIM** | 123140053 |
| **Kelas** | Deep Learning RB |
| **Mata Kuliah** | Pembelajaran Mendalam / Deep Learning (IF25-40401) |
| **Institusi** | Institut Teknologi Sumatera |
| **GitHub** | [@Ahmatsembiring](https://github.com/Ahmatsembiring) |

## Daftar Isi

- [Tentang Repositori](#tentang-repositori)
- [Struktur Repositori](#struktur-repositori)
- [Materi Perkuliahan](#materi-perkuliahan)
- [Latihan Model](#latihan-model)
- [Studio Proyek 1](#studio-proyek-1)
- [Studio Proyek 2](#studio-proyek-2)
- [Teknologi](#teknologi)
- [Cara Menjalankan](#cara-menjalankan)
- [Catatan](#catatan)

---

## Tentang Repositori

Repositori ini mendokumentasikan proses belajar saya di mata kuliah Pembelajaran Mendalam: materi perkuliahan, latihan praktikum, dan dua proyek studio. Seluruh implementasi menggunakan **PyTorch**, dengan alur topik sebagai berikut:

1. **Fondasi MLP** pada data tabular
2. **Transfer learning**, Batch Normalization, dan learning rate scheduler
3. **Diagnosis dan penanganan overfitting** (L2, Dropout, Early Stopping)
4. **ANN from scratch** pada CIFAR-10 dengan eksperimen optimizer, scheduler, regularisasi, dan kapasitas model
5. **Perbandingan 6 arsitektur CNN klasik** pada CIFAR-10

Repositori ini akan terus diperbarui seiring perkuliahan berjalan.

## Struktur Repositori

```text
Pembelajaran-Mendalam/
├── Modul/                           # Materi perkuliahan (PDF)
├── Latihan-Model/                   # Latihan praktikum Modul 1–3
│   ├── Model1/                      # Fondasi MLP
│   ├── Model2/                      # Transfer Learning & Konvergensi
│   └── Model3/                      # Diagnosis & Penanganan Overfitting
├── Tugas-Studio/                    # Studio Proyek 1: ANN from scratch pada CIFAR-10
│   ├── Tugas-Studio-1/              # Notebook, spesifikasi tugas, dan ekspor PDF
│   └── Tugas-studio-1-perbaikan/    # Versi perbaikan notebook
└── tugas-studio2/                   # Studio Proyek 2: Perbandingan 6 arsitektur CNN
```

## Materi Perkuliahan

- [Minggu 1 — Pengenalan Deep Learning](Modul/Mgg1%20-%20Pengenalan%20Deep%20Learning.pdf)
- [Minggu 2 — Training Deep Learning dan Transfer Learning](Modul/Mgg2%20-%20Training%20Deep%20Learning%20and%20Transfer%20Learning.pdf)
- [Modul 3 — Diagnosis dan Penanganan Overfitting](Modul/Modul03%20-%20Diagnosis%20dan%20Penanganan%20Overfitting.pdf)
- [Pertemuan 4 — CNN from Scratch](Modul/pertemuan-04-cnn-from-scratch.pdf)
- [Pertemuan 5 — CNN Variations](Modul/pertemuan-05-cnn-variations.pdf)

## Latihan Model

Latihan praktikum dari tiga modul awal, masing-masing dalam satu notebook.

| Modul | Topik | Hasil utama | Notebook |
|---|---|---|---|
| **1. Fondasi MLP** | MLP (8→16→8→1) untuk klasifikasi biner pada dataset *Pima Indians Diabetes*: arsitektur, inisialisasi bobot, loss function, optimizer, dan metrik evaluasi | Akurasi 72,73% · Precision 60,66% · Recall 67,27% | [Buka](Latihan-Model/Model1/Modul01_Fondasi_MLP.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmatsembiring/Pembelajaran-Mendalam/blob/main/Latihan-Model/Model1/Modul01_Fondasi_MLP.ipynb) |
| **2. Transfer Learning & Konvergensi** | Pre-training pada *FashionMNIST* (bobot disimpan sebagai `pretrained_fashion.pth`), lalu fine-tuning pada *MNIST* dengan Batch Normalization dan `ReduceLROnPlateau` | Akurasi uji 98,54% setelah 15 epoch fine-tuning | [Buka](Latihan-Model/Model2/Modul02_Transfer_Learning_Konvergensi.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmatsembiring/Pembelajaran-Mendalam/blob/main/Latihan-Model/Model2/Modul02_Transfer_Learning_Konvergensi.ipynb) |
| **3. Diagnosis & Penanganan Overfitting** | Overfitting sengaja diciptakan pada dataset sintetis (300 sampel, 20 fitur), lalu ditangani dengan L2 (weight decay), Dropout, dan Early Stopping | Akurasi uji 83,3% (overfit) → 84,2% (setelah penanganan). Early stopping terpicu di epoch 25, performa terbaik di epoch 5 | [Buka](Latihan-Model/Model3/Modul03_Diagnosis_Penanganan_Overfitting.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmatsembiring/Pembelajaran-Mendalam/blob/main/Latihan-Model/Model3/Modul03_Diagnosis_Penanganan_Overfitting.ipynb) |

## Studio Proyek 1

**Fine-Tuning ANN (from scratch) pada CIFAR-10**

Membangun ANN/MLP dari nol (tanpa CNN) untuk klasifikasi CIFAR-10 (60.000 citra berwarna 32×32 piksel, 10 kelas). Model baseline berarsitektur 3072 → 512 → 256 → 10, dilatih dengan augmentasi `RandomCrop` dan `RandomHorizontalFlip` pada data latih, serta Early Stopping dan Model Checkpointing yang diimplementasikan manual berdasarkan validation loss. Eksperimen dijalankan terkontrol: satu variabel diubah pada setiap percobaan.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmatsembiring/Pembelajaran-Mendalam/blob/main/Tugas-Studio/Tugas-Studio-1/CIFAR10_ANN_StudioProject1.ipynb)

**Hasil eksperimen** (validation set, satu kali eksekusi):

| Eksperimen | Konfigurasi | Val. Accuracy | Val. Loss | Waktu latih |
|---|---|---|---|---|
| Baseline | Adam, hidden layer 512-256 | 46,92% | 1,515 | 179 dtk (11 epoch) |
| SGD + Momentum | Optimizer diganti SGD + Momentum | 51,72% | 1,380 | 310 dtk (19 epoch) |
| StepLR | Adam + scheduler StepLR | 53,66% | 1,338 | 487 dtk (30 epoch) |
| Dropout | Dropout 0,3 | 44,96% | 1,562 | 466 dtk (28 epoch) |
| **BatchNorm** | Batch Normalization | **57,90%** | **1,201** | 513 dtk (30 epoch) |
| Dropout + BatchNorm | Dropout 0,3 + Batch Normalization | 53,76% | 1,299 | 507 dtk (30 epoch) |
| Kapasitas kecil | Hidden layer 256-128 (0,82 jt parameter) | 50,36% | 1,427 | 498 dtk (30 epoch) |
| Kapasitas besar | Hidden layer 1024-512-256 (3,81 jt parameter) | 48,54% | 1,451 | 368 dtk (22 epoch) |

**Temuan singkat:**

- Batch Normalization memberi hasil validasi terbaik (57,90%), sekitar 11 poin di atas baseline (46,92%).
- StepLR (53,66%) dan SGD + Momentum (51,72%) sama-sama melampaui baseline.
- Dropout 0,3 yang berdiri sendiri lebih rendah dari baseline (44,96%), sedangkan kombinasinya dengan Batch Normalization mencapai 53,76%.
- Model berkapasitas kecil (50,36%) sedikit lebih baik daripada model berkapasitas besar (48,54%).

**Berkas:**

- [`CIFAR10_ANN_StudioProject1.ipynb`](Tugas-Studio/Tugas-Studio-1/CIFAR10_ANN_StudioProject1.ipynb): notebook eksperimen utama (8 eksperimen, lengkap dengan output)
- [`STUDIO1_Kelompokxx_Ahmat_NamaAnggota2.ipynb`](Tugas-Studio/Tugas-Studio-1/STUDIO1_Kelompokxx_Ahmat_NamaAnggota2.ipynb): notebook yang disusun mengikuti poin-poin spesifikasi tugas
- [`Spesifikasi Tugas Kelompok - Studio Project 1.pdf`](Tugas-Studio/Tugas-Studio-1/Spesifikasi%20Tugas%20Kelompok%20-%20Studio%20Project%201.pdf): spesifikasi tugas
- [`vertopal.com_CIFAR10_ANN_StudioProject1 (1).pdf`](Tugas-Studio/Tugas-Studio-1/vertopal.com_CIFAR10_ANN_StudioProject1%20%281%29.pdf): ekspor PDF notebook
- [`Tugas-studio-1-perbaikan/STUDIO1_ANN_CIFAR10.ipynb`](Tugas-Studio/Tugas-studio-1-perbaikan/STUDIO1_ANN_CIFAR10.ipynb): versi perbaikan notebook

## Studio Proyek 2

**Eksplorasi, Perbandingan, dan Analisis 6 Arsitektur CNN pada CIFAR-10**

Dikerjakan berkelompok bersama **Tengku Hafid Diraputra (123140043)**.

Enam arsitektur CNN klasik diimplementasikan dan dilatih dari nol pada CIFAR-10, lalu dibandingkan dari sisi akurasi, jumlah parameter, komputasi, ukuran model, dan waktu pelatihan. Arsitektur yang dipakai adalah LeNet-5, AlexNet, VGG-16, GoogLeNet, serta dua model pilihan: ResNet-18 dan MobileNetV2.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmatsembiring/Pembelajaran-Mendalam/blob/main/tugas-studio2/Tugas_CNN_CIFAR10_6_Arsitektur.ipynb)

**Spesifikasi eksperimen** (sama untuk semua model):

| Ketentuan | Nilai |
|---|---|
| Dataset | CIFAR-10, pembagian default torchvision (50.000 citra latih, 10.000 citra uji) |
| Random seed | 42 |
| Batch size | 64 |
| Optimizer | Adam, learning rate 0,001 |
| Epoch | 10 |
| Metrik | Akurasi pada data uji |
| Kondisi | Dilatih dari nol (tanpa pretrained) pada resolusi 32×32, tanpa augmentasi dan tanpa scheduler |

**Hasil** (diurutkan berdasarkan akurasi uji pada epoch ke-10):

| Peringkat | Model | Tahun | Parameter | Akurasi uji | Waktu/epoch |
|---|---|---|---|---|---|
| 1 | **GoogLeNet** | 2014 | 5,87 juta | **83,98%** | 87,6 dtk |
| 2 | ResNet-18 | 2015 | 11,17 juta | 82,64% | 24,5 dtk |
| 3 | MobileNetV2 | 2018 | 2,24 juta | 80,90% | 30,5 dtk |
| 4 | VGG-16 | 2014 | 33,65 juta | 79,64% | 24,6 dtk |
| 5 | AlexNet | 2012 | 28,71 juta | 66,60% | 15,4 dtk |
| 6 | LeNet-5 | 1998 | 0,06 juta | 57,43% | 2,9 dtk |

**Temuan singkat:**

- Akurasi tidak sebanding lurus dengan jumlah parameter: AlexNet (28,7 juta parameter) hanya mencapai 66,60%, sedangkan MobileNetV2 (2,24 juta) mencapai 80,90%. Rancangan arsitektur (BatchNorm, Global Average Pooling, bottleneck 1×1, residual connection) lebih menentukan.
- ResNet-18 paling seimbang secara keseluruhan: sekitar 98% akurasi GoogLeNet dengan sekitar 38% komputasi (MAC) dan waktu per epoch 3,6× lebih cepat.
- Tanpa augmentasi data, model besar mulai overfitting (selisih akurasi latih dan uji pada ResNet-18 mencapai 14,33 poin), sedangkan VGG-16 masih membaik di akhir sehingga 10 epoch belum cukup baginya.

**Berkas:**

- [`Tugas_CNN_CIFAR10_6_Arsitektur.ipynb`](tugas-studio2/Tugas_CNN_CIFAR10_6_Arsitektur.ipynb): implementasi, pelatihan, dan analisis keenam arsitektur
- [`langkah1_pipeline_cifar10.ipynb`](tugas-studio2/langkah1_pipeline_cifar10.ipynb): pipeline data dan training loop (data dimuat langsung ke GPU, mixed precision/AMP)
- [`Studi Proyek 2 - 123140043 123140053.pdf`](tugas-studio2/Studi%20Proyek%202%20-%20123140043%20123140053.pdf): laporan lengkap (34 halaman)

## Teknologi

| Kategori | Tools |
|---|---|
| Bahasa | Python |
| Deep learning | PyTorch, torchvision |
| Data & ML | NumPy, Pandas, scikit-learn |
| Visualisasi | Matplotlib, Seaborn |
| Lingkungan | Jupyter Notebook, Google Colab (GPU T4) |

## Cara Menjalankan

### Opsi 1: Google Colab (disarankan)

1. Klik badge **Open In Colab** pada notebook yang ingin dijalankan.
2. Aktifkan GPU: **Runtime → Change runtime type → T4 GPU**.
3. Jalankan seluruh sel: **Runtime → Run all**.

Pada GPU T4, notebook Studio Proyek 2 memerlukan sekitar 35–50 menit hingga selesai.

### Opsi 2: Lokal

```bash
# 1. Clone repositori
git clone https://github.com/Ahmatsembiring/Pembelajaran-Mendalam.git
cd Pembelajaran-Mendalam

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 3. Pasang dependensi (disarankan PyTorch 2.x)
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter

# 4. Jalankan Jupyter
jupyter notebook
```

Dataset (MNIST, FashionMNIST, dan CIFAR-10) diunduh otomatis oleh `torchvision` saat notebook dijalankan, sedangkan dataset Pima Indians Diabetes pada Modul 1 diunduh dari internet, sehingga koneksi internet diperlukan.

## Catatan

- Repositori ini dibuat untuk dokumentasi dan pembelajaran pribadi. Silakan dijadikan referensi, tetapi mohon tidak disalin mentah-mentah untuk tugas perkuliahan.
- Hasil eksperimen dapat sedikit berbeda antar-eksekusi meskipun *random seed* sudah dikunci, tergantung perangkat (CPU/GPU) dan versi library.

---

<div align="center">

Dibuat oleh **Ahmat Prayoga Sembiring** (123140053) · Institut Teknologi Sumatera

Jika repositori ini bermanfaat, jangan ragu memberi ⭐

</div>
