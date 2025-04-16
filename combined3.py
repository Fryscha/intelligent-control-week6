import cv2
import numpy as np
from ultralytics import YOLO  # Pastikan ultralytics sudah terinstal

# Path ke model dan gambar input
MODEL_PATH = "best2.pt"
IMAGE_PATH = "dataset2.jpg"  # Ganti dengan path gambar yang ingin kamu proses

# Load model
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

def detect_canny_edges_with_model_from_image(image_path):
    # Load gambar
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Gambar tidak ditemukan atau tidak dapat dibuka.")
        return

    # Canny Edge Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Konversi ke RGB untuk YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prediksi YOLO
    results = model(frame_rgb)
    predictions = results[0]

    # Mask kosong untuk area deteksi
    mask_model = np.zeros_like(edges)
    overlay = frame.copy()

    # Jika model segmentasi
    if predictions.masks is not None and predictions.masks.xy:
        for mask in predictions.masks.xy:
            mask = np.array(mask, np.int32)
            cv2.fillPoly(overlay, [mask], (255, 0, 0))  # Biru
            cv2.polylines(frame, [mask], True, (0, 255, 0), 2)
            cv2.fillPoly(mask_model, [mask], 255)

    # Jika model bounding box
    elif predictions.boxes is not None and len(predictions.boxes.xyxy) > 0:
        for i, box in enumerate(predictions.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(mask_model, (x1, y1), (x2, y2), 255, -1)

            if hasattr(predictions.boxes, "cls"):
                label = f"Objek {int(predictions.boxes.cls[i])}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Gabungkan overlay (area deteksi)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Ubah edge ke BGR
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Warnai tepi sesuai area deteksi
    edges_bgr[(mask_model == 255) & (edges == 255)] = [0, 0, 255]      # merah
    edges_bgr[(mask_model == 0) & (edges == 255)] = [255, 255, 255]    # putih

    # Gabungkan hasil
    combined = np.hstack((frame, edges_bgr))

    # Simpan dan tampilkan
    cv2.imwrite("combined3_result.jpg", combined)
    cv2.imshow("Deteksi Gambar - YOLO + Canny", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan deteksi dari gambar
detect_canny_edges_with_model_from_image(IMAGE_PATH)