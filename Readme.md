# Laporan Proyek Machine Learning - Rangga Julian Syaputra

## Domain Proyek

### A. Latar Belakang

Pertumbuhan populasi, peningkatan urbanisasi, serta perkembangan infrastruktur yang pesat di wilayah California telah membawa dampak yang signifikan terhadap dinamika pasar perumahan. California, sebagai salah satu negara bagian dengan tingkat kepadatan penduduk tinggi di Amerika Serikat, menghadapi tantangan besar dalam penyediaan hunian yang layak dan terjangkau bagi masyarakat. Akibat dari tingginya permintaan dan keterbatasan lahan, harga rumah cenderung meningkat dan fluktuatif dari waktu ke waktu.

Selain itu, terdapat berbagai faktor yang mempengaruhi harga rumah, mulai dari variabel demografis seperti tingkat pendapatan, kepadatan penduduk, hingga faktor geografis seperti lokasi terhadap garis pantai (proximity to the ocean), koordinat geografis, dan usia bangunan. Kompleksitas dari berbagai variabel ini menuntut adanya pendekatan yang lebih sistematis dan berbasis data dalam memprediksi harga rumah, sehingga stakeholder seperti agen properti, investor, pengembang, dan konsumen akhir dapat mengambil keputusan yang lebih tepat.

Di era transformasi digital, pendekatan konvensional dalam menilai harga properti mulai tergantikan oleh pendekatan berbasis data science dan machine learning. Algoritma pembelajaran mesin dapat membantu dalam membangun sistem prediksi yang mampu mengakomodasi kompleksitas data dan menghasilkan estimasi harga yang lebih akurat. Dengan pemanfaatan algoritma regresi seperti Random Forest, sistem prediksi harga rumah dapat dikembangkan secara efisien untuk mendukung proses pengambilan keputusan di sektor properti.

### B. Permasalahan

Proyek ini berfokus pada upaya untuk menyelesaikan permasalahan utama yang dihadapi dalam dunia properti, khususnya dalam prediksi harga rumah. Permasalahan yang akan dianalisis antara lain:

- Bagaimana memanfaatkan data demografis dan geografis untuk membangun model prediksi harga rumah yang akurat?
- Algoritma machine learning mana yang paling efektif dalam menghasilkan prediksi harga rumah dengan kesalahan minimum?
- Bagaimana cara meningkatkan kinerja model dasar (baseline) melalui proses tuning hyperparameter atau penerapan model yang lebih kompleks?

Masalah-masalah ini menjadi penting untuk dijawab mengingat pengambilan keputusan yang lebih akurat dalam investasi properti sangat bergantung pada keandalan estimasi harga.

### C. Tujuan

Tujuan dari proyek ini secara umum adalah membangun solusi berbasis machine learning untuk prediksi harga rumah di California. Secara khusus, tujuan proyek ini meliputi:

1. Membangun model prediksi harga rumah menggunakan dataset California Housing yang memuat informasi geografis, demografis, dan karakteristik rumah.
2. Melakukan eksplorasi data dan preprocessing untuk memastikan kualitas data yang optimal dalam proses pelatihan model.
3. Menerapkan model baseline menggunakan algoritma Random Forest Regression.
4. Melakukan evaluasi kinerja model berdasarkan metrik evaluasi seperti Root Mean Squared Error (RMSE) dan R² Score.
5. Melakukan tuning hyperparameter untuk meningkatkan akurasi model.
6. Menyajikan interpretasi hasil dan insight bisnis yang relevan berdasarkan output model prediksi.

### D. Referensi

- California Housing Dataset (Kaggle). Tautan: [https://www.kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.)_. Sebastopol, CA: O’Reilly Media.
- Friedman, J., Hastie, T., & Tibshirani, R. (2001). _The Elements of Statistical Learning: Data Mining, Inference, and Prediction_. Springer Series in Statistics.
- Scikit-learn Documentation. (2024). _Random Forest Regressor_. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

## II. Business Understanding

### A. Pernyataan Masalah (Problem Statement)

Harga rumah merupakan variabel yang sangat kompleks dan dipengaruhi oleh berbagai faktor, baik demografis maupun geografis, yang saling berinteraksi secara dinamis. Faktor-faktor seperti lokasi, kondisi lingkungan sekitar, ukuran dan tipe rumah, serta karakteristik sosial-ekonomi penduduk dapat memengaruhi nilai sebuah properti. Oleh karena itu, memprediksi harga rumah secara akurat membutuhkan pendekatan yang komprehensif dan cermat.

Dalam konteks pasar properti di California yang sangat kompetitif dan berfluktuasi, prediksi harga rumah yang tepat sangat penting untuk membantu agen properti, investor, pembeli, dan pengembang dalam membuat keputusan yang lebih tepat dan strategis. Pendekatan machine learning menjadi solusi potensial yang mampu menangani kompleksitas data dan menghasilkan estimasi harga yang efektif dan efisien.

---

### B. Tujuan Proyek

Proyek ini bertujuan untuk:

- Mengembangkan sistem prediksi harga rumah di wilayah California dengan memanfaatkan data historis dan fitur relevan.
- Mengurangi tingkat kesalahan prediksi dengan mengimplementasikan algoritma machine learning yang sesuai dan optimal.
- Memberikan wawasan atau insight yang mendalam mengenai fitur-fitur utama yang secara signifikan memengaruhi harga rumah, sehingga dapat mendukung pengambilan keputusan yang lebih baik.

---

### C. Pernyataan Solusi (Solution Statement)

Dalam proyek ini, dua pendekatan solusi utama diusulkan untuk mengatasi permasalahan prediksi harga rumah:

1. **Baseline Model:**
   - Membangun model awal menggunakan algoritma **Random Forest Regressor** tanpa optimasi parameter sebagai tolok ukur dasar performa prediksi.
2. **Improved Model:**
   - Melakukan **hyperparameter tuning** menggunakan teknik **GridSearchCV** pada model Random Forest Regressor untuk mencari kombinasi parameter terbaik yang dapat meningkatkan akurasi prediksi sekaligus mengurangi kesalahan.

Kedua model tersebut akan dievaluasi dan dibandingkan menggunakan metrik evaluasi seperti **Root Mean Squared Error (RMSE)** dan **R² Score** untuk memastikan peningkatan kualitas prediksi.

## III. Data Understanding

### A. Deskripsi Dataset

Dataset yang digunakan dalam proyek ini adalah _California Housing Dataset_ yang diperoleh dari [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Dataset ini berisi informasi mengenai karakteristik geografis, demografis, dan sosial-ekonomi dari wilayah-wilayah di California, dengan tujuan utama memprediksi harga median rumah.

- **Jumlah data:** 20.640 baris (observasi)
- **Jumlah fitur:** 10 kolom (variabel)
- **Variabel target:** `median_house_value` (harga median rumah)
- **Sumber data:** [Kaggle - California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

---

### B. Struktur dan Tipe Data

Berdasarkan output `df.info()`:

- Kolom numerik berjumlah 9 dan bertipe `float64`, sedangkan kolom `ocean_proximity` bertipe `object` (kategori).
- Semua kolom memiliki 20.640 data non-null, kecuali `total_bedrooms` yang hanya memiliki 20.433 data non-null (terdapat missing values).
- Dataset memiliki memori sebesar ±1.6 MB.

---

### C. Statistik Deskriptif

| Fitur              | Mean      | Std Dev   | Min     | 25%      | 50%      | 75%      | Max      |
| ------------------ | --------- | --------- | ------- | -------- | -------- | -------- | -------- |
| longitude          | -119.57   | 2.00      | -124.35 | -121.80  | -118.49  | -118.01  | -114.31  |
| latitude           | 35.63     | 2.14      | 32.54   | 33.93    | 34.26    | 37.71    | 41.95    |
| housing_median_age | 28.63     | 12.59     | 1.0     | 18.0     | 29.0     | 37.0     | 52.0     |
| total_rooms        | 2635.76   | 2181.62   | 2.0     | 1447.75  | 2127.0   | 3148.0   | 39320.0  |
| total_bedrooms     | 537.87    | 421.39    | 1.0     | 296.0    | 435.0    | 647.0    | 6445.0   |
| population         | 1425.48   | 1132.46   | 3.0     | 787.0    | 1166.0   | 1725.0   | 35682.0  |
| households         | 499.54    | 382.33    | 1.0     | 280.0    | 409.0    | 605.0    | 6082.0   |
| median_income      | 3.87      | 1.90      | 0.50    | 2.56     | 3.53     | 4.74     | 15.00    |
| median_house_value | 206855.82 | 115395.62 | 14999.0 | 119600.0 | 179700.0 | 264725.0 | 500001.0 |

---

### D. Kondisi Data

#### 1. Missing Values

Terdapat nilai yang hilang (missing values) pada kolom `total_bedrooms` sebanyak **207** baris. Kolom lainnya tidak memiliki data yang hilang.

| Kolom          | Missing Values |
| -------------- | -------------- |
| total_bedrooms | 207            |
| lainnya        | 0              |

#### 2. Duplikat

Berdasarkan pemeriksaan dengan `df.duplicated().sum()`, tidak ditemukan data duplikat pada dataset ini.

#### 3. Outlier

Outlier terdeteksi pada beberapa fitur numerik, terutama:

- **`total_rooms`**, **`total_bedrooms`**, **`population`**, dan **`households`** memiliki nilai maksimum yang sangat jauh dari nilai kuartil ketiga (Q3), mengindikasikan kehadiran nilai-nilai ekstrem.
  - Misalnya, `total_rooms` memiliki Q3 sebesar **3.148**, namun maksimum mencapai **39.320**, hampir **12 kali lipat**. Ini sangat mungkin menjadi outlier.
  - `population` memiliki nilai maksimum **35.682**, yang juga jauh dari rata-rata **1.425** dan Q3 **1.725**.
- **`median_house_value`** memiliki nilai maksimum **500001**, dan ini bukan nilai alami tertinggi, tetapi hasil dari pembatasan (_capping_) dalam dataset. Artinya, harga-harga rumah di atas nilai ini tidak direpresentasikan secara proporsional, menyebabkan distribusi target menjadi tidak normal.
- Kehadiran outlier seperti ini dapat mengganggu hasil pelatihan model machine learning, khususnya model yang sensitif terhadap skala seperti regresi linear. Oleh karena itu, pendekatan seperti transformasi log, normalisasi, dan pemangkasan (_trimming_ atau _winsorizing_) dapat dipertimbangkan dalam tahap preprocessing.

---

### D. Eksplorasi dan Visualisasi Data

#### 1. Histogram Distribusi Fitur Numerik

![Visualisasi Distribusi Variabel Numerik](https://github.com/user-attachments/assets/77c567cf-5658-4dd2-8ede-535f7df7596c)

Menampilkan distribusi dari fitur numerik pada dataset:

- **`longitude` dan `latitude`**: Menunjukkan penyebaran lokasi geografis di California.
- **`housing_median_age`**: Sebagian besar rumah berusia antara 15–50 tahun.
- **`total_rooms`, `total_bedrooms`, `population`, `households`**: Memiliki distribusi yang sangat miring ke kanan (right-skewed), menunjukkan keberadaan nilai ekstrem yang besar.
- **`median_income`**: Terdistribusi normal condong ke kiri, mayoritas penduduk memiliki pendapatan rendah hingga menengah.
- **`median_house_value`**: Terlihat ada batas maksimum di $500.000, menunjukkan kemungkinan adanya capping pada data.

---

#### 2. Distribusi Kategori Ocean Proximity

![Visualisasi Kategori Ocean Proximity](https://github.com/user-attachments/assets/a9cfef5a-3207-4a1e-b0ab-f307ea3b0276)

Distribusi jumlah data berdasarkan kategori `ocean_proximity`:

- **`<1H OCEAN`** dan **`INLAND`** merupakan kategori dominan.
- Kategori seperti `ISLAND` sangat jarang muncul, kemungkinan merupakan outlier.
- Kategori ini penting dalam memodelkan harga rumah berdasarkan jaraknya terhadap laut.

---

#### 3. Peta Sebaran Lokasi Rumah Berdasarkan Harga

![Scatter Plot Lokasi dan Harga Rumah](https://github.com/user-attachments/assets/8aa3ec7a-b97e-4762-9b3a-ccb531c086d0)

Scatter plot `latitude vs longitude` dengan warna yang menunjukkan nilai `median_house_value`:

- Harga rumah yang tinggi terkonsentrasi di wilayah pesisir seperti Los Angeles dan San Francisco.
- Wilayah INLAND cenderung memiliki harga rumah lebih rendah.

---

#### 4. Top 10 Feature Importance

![Visualisasi Feature Importance](https://github.com/user-attachments/assets/e29d097c-c7fe-4575-b3a6-fe0280121955)

Hasil dari model (kemungkinan Decision Tree/Random Forest) menunjukkan fitur-fitur paling penting dalam memprediksi harga rumah:

- **`median_income`** adalah fitur paling penting.
- Fitur lokasi (`ocean_proximity_INLAND`, `longitude`, `latitude`) juga sangat berpengaruh.
- Fitur jumlah kamar (`total_rooms`, `total_bedrooms`) dan `housing_median_age` juga memberikan kontribusi meskipun lebih kecil.

---

#### 5. Prediksi vs Nilai Aktual

![Plot Prediksi vs Aktual](https://github.com/user-attachments/assets/2690496e-670c-4538-8a59-669a17a2fc3f)

Plot antara hasil prediksi model dan nilai aktual `median_house_value`:

- Titik-titik yang sejajar dengan garis merah (y = x) menunjukkan prediksi yang akurat.
- Sebagian besar prediksi cukup baik, namun terdapat penyebaran lebih besar pada nilai tinggi.

---

#### 6. Distribusi Residual Error

![Distribusi Residual Error](https://github.com/user-attachments/assets/0e616810-4b18-4edc-ab24-f5410efe6143)

Histogram dari nilai residual (selisih antara nilai aktual dan prediksi):

- Terdistribusi hampir normal dan simetris di sekitar nol.
- Menunjukkan bahwa model tidak memiliki bias besar terhadap over-prediksi atau under-prediksi.

---

#### 7. Residual Error vs Prediksi

![Residual Error vs Prediksi](https://github.com/user-attachments/assets/d42fc680-23f6-476c-88ea-a81a1ddedf27)

Plot residual terhadap nilai prediksi:

- Pola berbentuk segitiga mengindikasikan adanya **heteroskedastisitas** – error meningkat seiring dengan meningkatnya nilai prediksi.
- Hal ini menunjukkan bahwa model memiliki kesulitan dalam memprediksi harga rumah yang sangat tinggi, dan mungkin perlu dilakukan transformasi logaritmik pada target.

---

## IV. Data Preparation

### A. Proses yang Dilakukan

1. **Imputasi Nilai Kosong**

   - Pada kolom `total_bedrooms` terdapat data yang hilang (missing values). Untuk mengatasi hal ini, dilakukan imputasi dengan menggantikan nilai kosong menggunakan nilai median dari kolom tersebut. Pendekatan ini dipilih agar nilai imputasi tidak terpengaruh oleh outlier ekstrem.

2. **Normalisasi Fitur Numerik**

   - Seluruh fitur numerik, seperti `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, dan `median_income`, dinormalisasi menggunakan metode **StandardScaler**. Proses ini bertujuan untuk menyesuaikan skala fitur sehingga memiliki distribusi dengan rata-rata 0 dan standar deviasi 1, yang penting untuk meningkatkan performa algoritma machine learning yang sensitif terhadap perbedaan skala data.

3. **Encoding Fitur Kategorikal**
   - Fitur kategorikal `ocean_proximity` diubah menjadi representasi numerik menggunakan teknik **One-Hot Encoding**. Dengan demikian, setiap kategori akan menjadi kolom biner terpisah yang memudahkan algoritma dalam memahami dan mengolah data tersebut.

---

### B. Alasan Data Preparation

- **Imputasi Median:**  
  Metode imputasi median dipilih karena median merupakan ukuran yang tahan terhadap nilai ekstrem (outlier), sehingga mengurangi risiko bias yang dapat muncul jika menggunakan nilai rata-rata.

- **Normalisasi Data:**  
  Normalisasi sangat diperlukan terutama untuk algoritma seperti Random Forest, Gradient Boosting, atau regresi linear, yang dapat mengalami penurunan performa bila fitur memiliki rentang nilai yang sangat berbeda.

- **Encoding Kategorikal:**  
  Algoritma machine learning secara umum tidak dapat langsung memproses fitur berbentuk string. Oleh karena itu, proses encoding kategorikal sangat penting untuk mengubah data tersebut menjadi format numerik yang dapat dipahami oleh model.

## V. Modeling

### A. Algoritma yang Digunakan

Model yang digunakan dalam proyek ini adalah **Random Forest Regressor**, yaitu sebuah algoritma pembelajaran ensemble yang menggabungkan banyak pohon keputusan (decision tree) untuk meningkatkan akurasi prediksi. Random Forest bekerja dengan membangun sejumlah pohon pada data pelatihan dan menghasilkan prediksi akhir berdasarkan rata-rata output dari masing-masing pohon.

Algoritma ini sangat efektif untuk masalah regresi karena dapat menangani hubungan non-linear, fitur numerik dalam jumlah besar, serta cukup tahan terhadap overfitting. Hal ini menjadikannya pilihan yang kuat dalam prediksi nilai kontinyu.

### B. Tahapan dan Parameter dalam Modeling

Proses modeling dilakukan melalui dua tahap utama:

#### 1. Baseline Model

Model awal dibangun menggunakan pengaturan default dari Random Forest. Tujuannya adalah untuk memperoleh gambaran performa dasar sebelum dilakukan optimasi. Parameter yang digunakan dalam baseline model adalah sebagai berikut:

- Jumlah pohon (n_estimators): 100
- Nilai random_state: 42 (untuk memastikan replikasi hasil)
- Parameter lain mengikuti default dari pustaka scikit-learn

```python
# Cell 8: Training Model Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_prep, y_train)
```

Model ini dilatih menggunakan data training yang telah diproses, lalu dievaluasi performanya menggunakan data testing untuk mengetahui kemampuan awal model dalam melakukan prediksi.

#### 2. Improved Model (Hyperparameter Tuning)

Setelah membangun model awal, tahap selanjutnya adalah melakukan tuning terhadap beberapa parameter penting untuk mendapatkan performa terbaik. Proses ini dilakukan menggunakan teknik **Grid Search** dengan **cross-validation** 3-fold.

Parameter yang disesuaikan antara lain:

- **n_estimators**: jumlah pohon yang dibangun (contoh: 50, 100, 200)
- **max_depth**: kedalaman maksimum tiap pohon (contoh: 10, 20, 30, None)
- **min_samples_split**: jumlah minimum sampel untuk membagi node (contoh: 2, 5, 10)
- **min_samples_leaf**: jumlah minimum sampel di daun pohon (contoh: 1, 2, 4)
- **max_features**: jumlah fitur yang dipertimbangkan untuk split (contoh: sqrt, log2, None)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Hyperparameter Tuning dengan GridSearchCV

# Parameter grid yang diperluas
param_grid = {
    'n_estimators': [50, 100, 200],               # jumlah pohon
    'max_depth': [None, 10, 20, 30],              # kedalaman pohon
    'min_samples_split': [2, 5, 10],              # minimal sampel untuk split
    'min_samples_leaf': [1, 2, 4],                # minimal sampel di daun
    'max_features': ['sqrt', 'log2', None]        # fitur maksimum yang digunakan
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
    refit=True
)

# Fit ke data training
grid_search.fit(X_train_prep, y_train)

# Hasil tuning
print("Best Parameters:", grid_search.best_params_)
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))

# Simpan model terbaik
best_model = grid_search.best_estimator_
```

Tujuan dari proses tuning ini adalah menemukan kombinasi parameter yang menghasilkan performa terbaik, yaitu:

**Best Parameters:**
{'max_depth': 30, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}

Proses tuning ini juga bertujuan menjaga kemampuan generalisasi model agar tidak mengalami overfitting terhadap data training.

### C. Kelebihan dan Kekurangan Random Forest

#### Kelebihan

- **Robust terhadap overfitting**: karena menggunakan rata-rata hasil dari banyak pohon, Random Forest lebih tahan terhadap overfitting dibandingkan decision tree tunggal.
- **Tidak memerlukan scaling fitur**: karena berbasis pohon, algoritma ini tidak terpengaruh oleh perbedaan skala antar fitur.
- **Dapat menangani data kompleks**: cocok untuk dataset dengan fitur dalam jumlah banyak dan korelasi non-linear antar fitur.
- **Memberikan informasi feature importance**: membantu dalam proses interpretasi dengan menunjukkan fitur mana yang paling berpengaruh terhadap prediksi.

#### Kekurangan

- **Sulit diinterpretasikan**: meskipun dapat menunjukkan pentingnya fitur, model ini tetap termasuk dalam kategori “black-box” karena sulit dijelaskan secara rinci proses pengambilan keputusan.
- **Waktu pelatihan lebih lama**: terutama ketika jumlah pohon dan kedalaman pohon besar, serta ketika dilakukan tuning parameter.
- **Konsumsi memori tinggi**: karena menyimpan banyak pohon secara simultan, model ini membutuhkan lebih banyak memori dan daya komputasi dibandingkan model sederhana.

---

## VI. Evaluation

### A. Metrik Evaluasi

Evaluasi model dilakukan dengan dua metrik utama yang sering digunakan dalam regresi:

- **Root Mean Squared Error (RMSE)**  
  Metrik ini menunjukkan rata-rata kesalahan prediksi dalam satuan yang sama dengan target. Nilai RMSE yang lebih rendah menunjukkan performa model yang lebih baik.

- **R² Score (Coefficient of Determination)**  
  R² menunjukkan proporsi variansi pada variabel target yang dapat dijelaskan oleh model. Nilai R² berkisar dari 0 hingga 1. Semakin mendekati 1, semakin baik performa model dalam menjelaskan variasi data.

### B. Hasil Evaluasi

Tabel berikut menunjukkan hasil evaluasi model pada data testing:

| Model          | RMSE     | R² Score |
| -------------- | -------- | -------- |
| Baseline Model | 48941.95 | 0.8172   |
| Improved Model | 48800.70 | 0.8183   |

Hasil baseline model dapat dilihat pada **Cell 8.1** di notebook, dengan metrik RMSE sebesar 48941.95 dan R² sebesar 0.8172. Sedangkan hasil improved model, yang diperoleh setelah proses hyperparameter tuning, dapat ditemukan pada **Cell 10** di notebook dengan RMSE sebesar 48800.70 dan R² sebesar 0.8183.

### C. Analisis Hasil

Berdasarkan hasil evaluasi, model yang telah melalui proses tuning (improved model) menunjukkan sedikit peningkatan performa dibandingkan baseline. Meskipun selisih RMSE tidak terlalu besar, peningkatan ini menunjukkan bahwa tuning parameter dapat membantu model menghasilkan prediksi yang lebih akurat dan stabil.

Selain itu, nilai R² yang mencapai lebih dari 0.81 menunjukkan bahwa model cukup baik dalam menjelaskan variansi target, dan dapat digunakan untuk mendukung pengambilan keputusan berbasis data dalam konteks prediksi ini.

---

## VII. Kesimpulan dan Rekomendasi

### A. Kesimpulan

- Algoritma Random Forest Regressor efektif dalam memprediksi harga rumah pada dataset California Housing.
- Fitur `median_income` memberikan pengaruh paling signifikan terhadap prediksi harga.
- Proses hyperparameter tuning berhasil meningkatkan performa model dengan menurunkan RMSE dan menaikkan R².

### B. Rekomendasi

- Eksplorasi penggunaan algoritma lain seperti XGBoost atau LightGBM untuk potensi peningkatan akurasi.
- Integrasikan data eksternal tambahan, misalnya kualitas sekolah, tingkat kriminalitas, atau fasilitas umum yang dapat memengaruhi harga rumah.
- Pertimbangkan transformasi log pada variabel target (`median_house_value`) untuk mengatasi masalah skewness dan meningkatkan kestabilan model.

---

## VIII. Referensi

1. Kaggle. (2020). _California Housing Prices Dataset_.  
   https://www.kaggle.com/datasets/camnugent/california-housing-prices

2. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O’Reilly Media.

3. Friedman, J., Hastie, T., & Tibshirani, R. (2001). _The Elements of Statistical Learning_. Springer.

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
