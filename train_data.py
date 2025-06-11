import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- KONFIGURASI ---
# Ganti path ini dengan lokasi folder dataset 'train' Anda
dataset_path = r'Klasifikasi Bahasa Isyarat SIBI.v2i.folder/train' 
# Nama file untuk menyimpan model yang sudah dilatih
output_model_file = 'random_forest_model_corrected_2.pkl'

# Ukuran gambar yang konsisten dengan skrip deteksi
img_size = (64, 64)

# --- LANGKAH 1: MEMUAT GAMBAR DARI DATASET ---
print("--- Langkah 1: Memuat Gambar dari Dataset ---")
# Inisialisasi list untuk menyimpan gambar yang telah diproses dan labelnya
images = []
labels = []

# Periksa apakah path dataset ada
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Folder dataset tidak ditemukan di '{dataset_path}'")
    print("Pastikan path sudah benar.")
else:
    # Iterasi ke setiap subfolder (misalnya: 'A', 'B', 'C', ...)
    for label_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label_folder)
        
        if os.path.isdir(folder_path):
            print(f"Memproses folder: {label_folder}")
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                
                # Baca gambar
                img = cv2.imread(filepath)
                if img is None:
                    print(f"⚠️ Gagal membaca gambar: {filepath}")
                    continue

                # Resize dan konversi ke grayscale
                img_resized = cv2.resize(img, img_size)
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                
                # Tambahkan gambar dan label ke list
                images.append(img_gray)
                labels.append(label_folder)

    print(f"\n✅ Selesai memuat {len(images)} gambar.")

# --- LANGKAH 2: EKSTRAKSI FITUR HOG ---
# Ini adalah langkah paling penting yang hilang dari skrip Anda sebelumnya.
# Kita harus mengubah setiap gambar menjadi vektor fitur HOG.
print("\n--- Langkah 2: Mengekstrak Fitur HOG dari Setiap Gambar ---")
hog_features = []
for image in images:
    # Gunakan parameter HOG yang SAMA PERSIS dengan skrip deteksi Anda
    feature_vector = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    hog_features.append(feature_vector)

print("✅ Selesai mengekstrak fitur HOG.")

# --- LANGKAH 3: MEMPERSIAPKAN DATA UNTUK PELATIHAN ---
# Konversi list ke array NumPy agar bisa digunakan oleh Scikit-learn
X = np.array(hog_features)
y = np.array(labels)

# Cek apakah kita punya cukup data untuk dilatih
if len(set(y)) > 1 and len(X) > 0:
    print(f"\nBentuk data fitur (X): {X.shape}") # Seharusnya (jumlah_gambar, 1764)
    print(f"Bentuk data label (y): {y.shape}")

    # Bagi data menjadi data latih (train) dan data uji (test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dibagi menjadi {len(X_train)} sampel latih dan {len(X_test)} sampel uji.")

    # --- LANGKAH 4: MELATIH MODEL RANDOM FOREST ---
    print("\n--- Langkah 4: Melatih Model RandomForestClassifier ---")
    # Inisialisasi model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Latih model dengan data fitur HOG
    model.fit(X_train, y_train)
    print("✅ Model berhasil dilatih.")

    # --- LANGKAH 5: EVALUASI DAN SIMPAN MODEL ---
    print("\n--- Langkah 5: Mengevaluasi dan Menyimpan Model ---")
    # Prediksi pada data uji untuk mengukur akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model pada data uji: {accuracy * 100:.2f}%")

    # Simpan model yang sudah benar ke file
    joblib.dump(model, output_model_file)
    print(f"✅ Model telah disimpan sebagai '{output_model_file}'")
    print("\nAnda sekarang bisa menggunakan file model ini di skrip deteksi real-time Anda!")

else:
    print("\n❌ Tidak cukup data atau label untuk melanjutkan pelatihan.")
