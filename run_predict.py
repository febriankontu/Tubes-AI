import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

def draw_subtitle(frame, text):
    """
    Fungsi untuk menggambar teks di bagian bawah frame seperti subtitle.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255) # Putih
    bg_color = (0, 0, 0) # Hitam

    # Dapatkan ukuran frame dan teks
    (frame_h, frame_w) = frame.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Hitung posisi untuk subtitle
    # Buat latar belakang semi-transparan
    subtitle_bg = np.zeros((text_h + 20, frame_w, 3), dtype=np.uint8)
    # Gabungkan latar belakang dengan frame
    alpha = 0.5
    blended_roi = cv2.addWeighted(frame[frame_h - (text_h + 20):frame_h, 0:frame_w], alpha, subtitle_bg, 1 - alpha, 0)
    frame[frame_h - (text_h + 20):frame_h, 0:frame_w] = blended_roi

    # Tulis teks di tengah bawah
    text_x = (frame_w - text_w) // 2
    text_y = frame_h - 10 - baseline
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def run_detection():
    """
    Fungsi untuk menjalankan deteksi gestur secara real-time menggunakan webcam.
    """
    model_filename = 'random_forest_model_corrected.pkl'
    
    # Periksa apakah model ada. Jika tidak, program akan berhenti.
    if not os.path.exists(model_filename):
        print(f"Error: File model '{model_filename}' tidak ditemukan.")
        print("Pastikan Anda memiliki model yang sudah dilatih di folder yang sama dengan skrip ini.")
        return

    print(f"--- Memuat Model '{model_filename}' dan Menjalankan Deteksi ---")
    
    # Muat model RandomForest
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return
    
    # Pastikan model yang dimuat mengharapkan jumlah fitur yang benar
    expected_features = 1764 
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != expected_features:
        print(f"Peringatan: Model mengharapkan {model.n_features_in_} fitur, tetapi skrip ini menghasilkan {expected_features} fitur.")
        print("Ini adalah penyebab error. Model Anda harus dilatih ulang dengan benar.")

    cap = cv2.VideoCapture(0) # Gunakan 0 untuk webcam default, atau 1 untuk webcam eksternal
    if not cap.isOpened():
        print("Error: Tidak bisa membuka webcam.")
        return
        
    roi_size = (64, 64)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Area untuk deteksi tangan
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
        roi_frame = frame[100:400, 300:600]

        # Proses deteksi kulit
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detection_text = "Tangan tidak terdeteksi"

        if contours and len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)

            if cv2.contourArea(cnt) > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                # KOTAK PEMBATAS DIHAPUS SESUAI PERMINTAAN
                # cv2.rectangle(roi_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                hand_roi = roi_frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, roi_size)

                # Ekstraksi fitur HOG
                feature = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                feature = feature.reshape(1, -1)

                try:
                    # Prediksi gestur
                    pred = model.predict(feature)[0]
                    pred_proba = model.predict_proba(feature)[0]
                    confidence = max(pred_proba)
                    detection_text = f"Deteksi: {pred} ({confidence:.2f})"
                except Exception as e:
                    # Error akan tetap muncul di sini jika model salah
                    print(f"Error saat prediksi: {e}")
                    detection_text = "ERROR: Model tidak kompatibel!"

        # Tampilkan hasil sebagai subtitle
        draw_subtitle(frame, detection_text)

        cv2.imshow("Deteksi Tangan & Prediksi (q=quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()
