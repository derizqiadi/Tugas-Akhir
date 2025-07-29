import cv2
import threading
import time
import re
import math
import os
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import sys

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

# Hardware Libraries
try:
    import gpiod
    import board
    import busio
    from PIL import Image, ImageDraw, ImageFont
    import adafruit_ssd1306
    IS_HARDWARE_AVAILABLE = True
    print("? Library hardware (gpiod, adafruit) ditemukan. Mode hardware diaktifkan.")
except ImportError as e:
    IS_HARDWARE_AVAILABLE = False
    print(f"? Menjalankan dalam mode simulasi PC. Error: {str(e)}")

# Firebase Libraries
try:
    import firebase_admin
    from firebase_admin import credentials, db
    IS_FIREBASE_AVAILABLE = True
    print("? Library Firebase ditemukan.")
except ImportError as e:
    IS_FIREBASE_AVAILABLE = False
    print(f"? Fungsi database online nonaktif. Error: {str(e)}")

# SQLite
import sqlite3
from tracker import Tracker

# =============================================================================
# KONFIGURASI UTAMA
# =============================================================================
YOLO_MODEL_PATH = 'best.pt'
FIREBASE_CRED_PATH = 'testtugasakhir-firebase-adminsdk-fbsvc-6d5fff6fbe.json'
FIREBASE_DB_URL = 'https://testtugasakhir-default-rtdb.firebaseio.com/'
INPUT_MODE = 'WEBCAM'  # 'WEBCAM' atau 'VIDEO'
VIDEO_SOURCE_PATH = 'coba1.mp4'
CAMERA_SOURCE_IDX_MASUK = 0  # Kamera masuk
CAMERA_SOURCE_IDX_KELUAR = 2 # Kamera keluar
SAVE_CROP_DIR = 'crop_yolo'
OCR_LOG_FILE = 'log_ocr.txt'
YOLO_IMG_SZ, CONF_THRESHOLD = 160, 0.1  # Ukuran lebih kecil untuk performa
VIRTUAL_LINE_Y_MASUK, LINE_OFFSET = 251, 15
HORIZONTAL_EXPAND_PERCENT, VERTICAL_SHRINK_PERCENT = 0.08, 0.10
DETECTION_COOLDOWN = 1
LED_PIN, BUZZER_PIN, GPIO_CHIP = 17, 27, 'gpiochip4'
OLED_WIDTH, OLED_HEIGHT, OLED_STANDBY_DELAY = 128, 32, 4
TARGET_FPS = 15  # Frame rate target untuk kedua kamera
SKIP_FRAMES = 1  # Lewati frame untuk mengurangi beban
PREVIEW_WIDTH = 600  # Lebar maksimum preview
PREVIEW_HEIGHT = 300  # Tinggi maksimum preview

# Variabel global
last_successful_detection_time = 0
model = None
ocr = None
fb_ref = None
sqlite_conn = None
chip = None
led_line = None
buzzer_line = None
oled = None
font = None
font_kecil = None

# =============================================================================
# FUNGSI INISIALISASI
# =============================================================================
def initialize_components():
    global model, ocr, fb_ref, sqlite_conn, chip, led_line, buzzer_line, oled, font, font_kecil
    
    print("? Menginisialisasi komponen...")
    
    # Model YOLO
    model = YOLO(YOLO_MODEL_PATH)
    
    # PaddleOCR
    ocr = PaddleOCR(
    rec_model_dir='custom_model/rec',  
    det_model_dir=None,                # Nonaktifkan deteksi bawaan
    use_angle_cls=True,               
    lang='en',
    use_gpu=False,
    show_log=True,
    enable_mkldnn=False,                                 
    )
    
    # Firebase
    if IS_FIREBASE_AVAILABLE:
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            if not firebase_admin._apps: 
                firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
            fb_ref = db.reference('parkir')
            print("? Firebase berhasil diinisialisasi.")
        except Exception as e:
            globals()['IS_FIREBASE_AVAILABLE'] = False
            print(f"? Gagal inisialisasi Firebase: {e}")
    
    # SQLite
    sqlite_conn = sqlite3.connect('parkir_lokal.db', check_same_thread=False)
    with sqlite_conn:
        sqlite_conn.execute('''CREATE TABLE IF NOT EXISTS riwayat_parkir 
                            (id INTEGER PRIMARY KEY, plat_nomor TEXT, 
                             waktu_masuk TEXT, status TEXT)''')
    print("? SQLite berhasil diinisialisasi.")
    
    # Direktori penyimpanan
    if not os.path.exists(SAVE_CROP_DIR):
        os.makedirs(SAVE_CROP_DIR)
    if not os.path.exists(OCR_LOG_FILE):
        with open(OCR_LOG_FILE, 'w') as f:
            f.write("Log Pembacaan OCR\n" + "="*80 + "\n")
            f.write("Timestamp | ID | Status | Hasil Baca | Keyakinan | Fragmen | Latency (ms)\n")
    
    # Hardware
    if IS_HARDWARE_AVAILABLE:
        try:
            chip = gpiod.Chip(GPIO_CHIP)
            led_line = chip.get_line(LED_PIN)
            buzzer_line = chip.get_line(BUZZER_PIN)
            led_line.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
            buzzer_line.request(consumer="buzzer", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
            
            i2c = busio.I2C(board.SCL, board.SDA)
            oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_kecil = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            print("? Perangkat keras berhasil diinisialisasi.")
        except Exception as e:
            globals()['IS_HARDWARE_AVAILABLE'] = False
            print(f"? Gagal inisialisasi hardware: {e}")
    
    print("? Inisialisasi sistem selesai.")

# =============================================================================
# CLASS KAMERA THREAD 
# =============================================================================
class CameraThread(threading.Thread):
    def __init__(self, source_idx, is_masuk=True):
        threading.Thread.__init__(self)
        self.source_idx = source_idx
        self.is_masuk = is_masuk
        self.frame = None
        self.running = True
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_count = 0
        self.actual_fps = 0
        self.last_fps_calc_time = time.time()
        
    def run(self):
        self.cap = cv2.VideoCapture(self.source_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_interval = 1.0 / TARGET_FPS
        
        while self.running:
            start_time = time.time()
            
            # Baca frame dari kamera
            ret, frame = self.cap.read()
            if not ret:
                print(f"? Kamera {'MASUK' if self.is_masuk else 'KELUAR'} terputus.")
                break
                
            self.frame_count += 1
            current_time = time.time()
            
            # Hitung FPS aktual setiap 1 detik
            if current_time - self.last_fps_calc_time >= 1.0:
                self.actual_fps = self.frame_count / (current_time - self.last_fps_calc_time)
                self.frame_count = 0
                self.last_fps_calc_time = current_time
            
            # Skip frame jika diperlukan
            if SKIP_FRAMES > 0 and self.frame_count % (SKIP_FRAMES + 1) != 0:
                continue
                
            with self.lock:
                self.frame = frame.copy()
                self.last_frame_time = time.time()
            
            # Hitung waktu yang dibutuhkan dan tunggu sesuai interval
            processing_time = time.time() - start_time
            wait_time = max(0, frame_interval - processing_time)
            time.sleep(wait_time)
                
        if self.cap:
            self.cap.release()
            
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.running = False

# =============================================================================
# FUNGSI BANTUAN 
# =============================================================================
def show_separate_previews(original, processed):
    """Menampilkan gambar asli dan hasil preprocessing di jendela terpisah"""
    if original is None or processed is None:
        return
        
    try:
        # Tampilkan gambar asli (crop)
        h_orig, w_orig = original.shape[:2]
        scale_orig = min(PREVIEW_WIDTH/w_orig, PREVIEW_HEIGHT/h_orig)
        new_w_orig = int(w_orig * scale_orig)
        new_h_orig = int(h_orig * scale_orig)
        resized_orig = cv2.resize(original, (new_w_orig, new_h_orig))
        cv2.imshow("1. Original Crop", resized_orig)
        
        # Tampilkan gambar hasil preprocessing
        h_proc, w_proc = processed.shape[:2]
        scale_proc = min(PREVIEW_WIDTH/w_proc, PREVIEW_HEIGHT/h_proc)
        new_w_proc = int(w_proc * scale_proc)
        new_h_proc = int(h_proc * scale_proc)
        resized_proc = cv2.resize(processed, (new_w_proc, new_h_proc))
        cv2.imshow("2. Processed Result", resized_proc)
        
    except Exception as e:
        print(f"? Gagal menampilkan preview terpisah: {e}")

def preprocess_plate(plate_image):
    if plate_image is None or plate_image.size == 0:
        return None
    
    try:
        original = plate_image.copy()
        
        # 1. Upscale gambar
        h, w = plate_image.shape[:2]
        upscaled_image = cv2.resize(plate_image, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoise
        denoised_image = cv2.fastNlMeansDenoisingColored(upscaled_image, None, 10, 10, 7, 21)

        # 3. Contrast Enhancement
        lab = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        final_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Tampilkan preview terpisah
        show_separate_previews(original, final_image)
        
        return final_image
    except Exception as e:
        print(f"? Error dalam preprocessing plat: {e}")
        return None

def process_ocr_results(ocr_result):
    # Pengecekan awal 
    if not ocr_result or not ocr_result[0] or not ocr_result[0][0]:
        return {'valid_plate': None, 'raw_text': 'NO_RESULT', 'avg_confidence': 0.0, 'fragment_count': 0}

    try:
        
        line = ocr_result[0][0]
        
        if not isinstance(line, (tuple, list)) or len(line) < 2:
            return {'valid_plate': None, 'raw_text': 'BAD_FORMAT', 'avg_confidence': 0.0, 'fragment_count': 0}

        # Ekstrak teks dan skor keyakinan
        raw_text, avg_confidence = line
        
        # proses validasi plat nomor 
        cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', cleaned_text)
        
        if match:
            # Jika format plat valid, bentuk kembali dengan spasi
            valid_plate = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        else:
            valid_plate = None

        return {
            'valid_plate': valid_plate, 
            'raw_text': raw_text, 
            'avg_confidence': avg_confidence, 
            'fragment_count': 1 
        }
        
    except Exception as e:
        print(f"? Error dalam memproses hasil OCR: {e}")
        print(f"  -> OCR Raw Result yang menyebabkan error: {ocr_result}")
        return {'valid_plate': None, 'raw_text': 'ERROR', 'avg_confidence': 0.0, 'fragment_count': 0}

def handle_database_and_reset_oled(plate_number, status="Masuk"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plate_key = plate_number.replace(" ", "_")
    
    try:
        with sqlite3.connect('parkir_otomatis.db', timeout=10) as conn:
            cursor = conn.cursor()
            data = {'plat_nomor': plate_number, 'waktu_masuk': timestamp, 'status': status}
            
            if IS_FIREBASE_AVAILABLE:
                db.reference(f'parkir/{plate_key}').set(data)
            
            cursor.execute("INSERT INTO riwayat_parkir (plat_nomor, waktu_masuk, status) VALUES (?, ?, ?)", 
                         (plate_number, timestamp, status))
            conn.commit()
            print(f"    [DB] Data {status} untuk {plate_number} berhasil disimpan.")
    except Exception as e:
        print(f"? ERROR saat akses database: {e}")
    
    time.sleep(OLED_STANDBY_DELAY)
    update_oled("Sistem Siaga", "Mencari Plat...")

def update_oled(line1, line2=""):
    if not IS_HARDWARE_AVAILABLE: 
        print(f"[OLED SIM] L1: {line1} | L2: {line2}")
        return
    
    try:
        image = Image.new("1", (oled.width, oled.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
        draw.text((0, 0), line1, font=font, fill=255)
        draw.text((0, 16), line2, font=font_kecil, fill=255)
        oled.image(image)
        oled.show()
    except Exception as e:
        print(f"? Error update OLED: {e}")

def signal_event(success=True):
    if not IS_HARDWARE_AVAILABLE: 
        print(f"[EVENT SIM] Success: {success}")
        return
    
    try:
        if success:
            led_line.set_value(1)
            buzzer_line.set_value(1)
            time.sleep(0.5)
            led_line.set_value(0)
            buzzer_line.set_value(0)
        else:
            for _ in range(3):
                buzzer_line.set_value(1)
                time.sleep(0.1)
                buzzer_line.set_value(0)
                time.sleep(0.1)
    except Exception as e:
        print(f"? Error dalam signal event: {e}")

# =============================================================================
# FUNGSI UTAMA PROSES DETEKSI
# =============================================================================
def process_detection(frame, tracker, processed_ids, is_masuk=True):
    global last_successful_detection_time
    
    if frame is None or time.time() - last_successful_detection_time < DETECTION_COOLDOWN:
        return frame
    
    # Mulai menghitung waktu pemrosesan
    start_time = time.time()
    
    try:
        results = model.predict(frame, classes=[0], imgsz=YOLO_IMG_SZ, conf=CONF_THRESHOLD, verbose=False)
        plate_boxes = [list(map(int, box.xyxy[0])) for r in results for box in r.boxes]
        
        valid_plate_boxes = []
        for box in plate_boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if h > 0 and 2.0 < (w/h) < 4.5:
                valid_plate_boxes.append(box)
        
        bbox_idx = tracker.update(valid_plate_boxes)
        
        for x1, y1, x2, y2, obj_id in bbox_idx:
            cy = (y1 + y2) // 2
            line_y = VIRTUAL_LINE_Y_MASUK if is_masuk else (frame.shape[0] - VIRTUAL_LINE_Y_MASUK)
            
            if (line_y - LINE_OFFSET < cy < line_y + LINE_OFFSET):
                if obj_id not in processed_ids:
                    processed_ids.add(obj_id)
                    
                    box_w, box_h = x2 - x1, y2 - y1
                    expand_w = int(box_w * HORIZONTAL_EXPAND_PERCENT)
                    shrink_h = int(box_h * VERTICAL_SHRINK_PERCENT)
                    x1_new = max(0, x1 - expand_w)
                    y1_new = max(0, y1 + shrink_h)
                    x2_new = min(frame.shape[1], x2 + expand_w)
                    y2_new = min(frame.shape[0], y2 - shrink_h)
                    plate_crop = frame[y1_new:y2_new, x1_new:x2_new]
                    
                    if plate_crop.size > 0:
                        # Tampilkan preview crop dari kamera
                        try:
                            h_crop, w_crop = plate_crop.shape[:2]
                            scale_crop = min(PREVIEW_WIDTH/w_crop, PREVIEW_HEIGHT/h_crop)
                            resized_crop = cv2.resize(plate_crop, (int(w_crop*scale_crop), int(h_crop*scale_crop)))
                            cv2.imshow(f"0. Camera {'Masuk' if is_masuk else 'Keluar'} - Crop", resized_crop)
                        except Exception as e:
                            print(f"? Gagal menampilkan preview crop: {e}")
                        
                        ocr_input_image = preprocess_plate(plate_crop)
                        if ocr_input_image is None:
                            continue
                        
                        t_start = time.time()
                        try:
                            ocr_result = ocr.ocr(ocr_input_image, det=False, cls=True)
                        except Exception as e:
                            print(f"? Error dalam proses OCR: {e}")
                            continue
                        
                        ocr_latency_ms = (time.time() - t_start) * 1000
                        
                        ocr_metrics = process_ocr_results(ocr_result)
                        final_plate = ocr_metrics['valid_plate']
                        
                        if final_plate:
                            status = "MASUK" if is_masuk else "KELUAR"
                            print(f"? Plat terbaca ({status}): {final_plate}")
                            update_oled(final_plate, f"AKSES {status}")
                            signal_event(success=True)
                            threading.Thread(
                                target=handle_database_and_reset_oled,
                                args=(final_plate, status)
                            ).start()
                        else:
                            raw_text = ocr_metrics['raw_text']
                            print(f"    -> Gagal validasi. Teks Mentah: '{raw_text}'")
                            update_oled(raw_text if raw_text else "Gagal Baca", "TIDAK VALID")
                            signal_event(success=False)
                        
                        last_successful_detection_time = time.time()
                        
                        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        status_log = "SUKSES" if final_plate else "GAGAL"
                        hasil_terbaca = final_plate if final_plate else ocr_metrics['raw_text']
                        
                        with open(OCR_LOG_FILE, 'a') as f:
                            log_entry = (
                                f"[{log_time}] | ID:{obj_id:<4} | Status: {status_log:<7} | "
                                f"Hasil: {hasil_terbaca:<12} | Keyakinan: {ocr_metrics['avg_confidence']:.2%} | "
                                f"Fragmen: {ocr_metrics['fragment_count']} | Latency: {ocr_latency_ms:.0f} ms\n"
                            )
                            f.write(log_entry)
                        
                        if hasil_terbaca:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            safe_text = re.sub(r'[^A-Z0-9]', '', hasil_terbaca)
                            direction = "MASUK" if is_masuk else "KELUAR"
                            filename = f"{timestamp_str}_ID{obj_id}_{direction}_READ({safe_text}).jpg"
                            save_path = os.path.join(SAVE_CROP_DIR, filename)
                            try:
                                cv2.imwrite(save_path, ocr_input_image)
                                print(f"    -> Gambar pra-pemrosesan disimpan: {save_path}")
                            except Exception as e:
                                print(f"? Gagal menyimpan gambar: {e}")
        
        # Hitung FPS pemrosesan dan tampilkan di preview
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Gambar bounding box dan garis virtual
        for x1_orig, y1_orig, x2_orig, y2_orig, _ in bbox_idx:
            try:
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
            except Exception as e:
                print(f"? Gagal menggambar bounding box: {e}")
        
        line_y = VIRTUAL_LINE_Y_MASUK if is_masuk else (frame.shape[0] - VIRTUAL_LINE_Y_MASUK)
        try:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
            # Tampilkan FPS di preview
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"? Gagal menggambar garis atau teks FPS: {e}")
    
    except Exception as e:
        print(f"? Error dalam proses deteksi: {e}")
    
    return frame

# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    initialize_components()
    
    # Inisialisasi kamera
    cam_masuk = CameraThread(CAMERA_SOURCE_IDX_MASUK, is_masuk=True)
    cam_keluar = CameraThread(CAMERA_SOURCE_IDX_KELUAR, is_masuk=False)
    
    cam_masuk.start()
    cam_keluar.start()
    
    # Tunggu sampai kamera siap
    time.sleep(2)
    
    tracker_masuk = Tracker()
    tracker_keluar = Tracker()
    processed_ids_masuk = set()
    processed_ids_keluar = set()
    
    oled_status_is_event = False
    last_event_time = 0
    
    if IS_HARDWARE_AVAILABLE:
        update_oled("Sistem Siaga", "Mencari Plat...")
    
    print(f"? Memproses dari sumber kamera... Tekan 'q' untuk keluar.")
    
    try:
        while True:
            frame_masuk = cam_masuk.get_frame()
            frame_keluar = cam_keluar.get_frame()
            
            if frame_masuk is None and frame_keluar is None:
                print("? Kedua kamera tidak memberikan frame.")
                break
            
            display_frames = []
            
            # Proses frame masuk
            if frame_masuk is not None:
                processed_frame_masuk = process_detection(
                    frame_masuk, tracker_masuk, processed_ids_masuk, is_masuk=True)
                if processed_frame_masuk is not None:
                    display_frames.append(processed_frame_masuk)
            
            # Proses frame keluar
            if frame_keluar is not None:
                processed_frame_keluar = process_detection(
                    frame_keluar, tracker_keluar, processed_ids_keluar, is_masuk=False)
                if processed_frame_keluar is not None:
                    display_frames.append(processed_frame_keluar)
            
            # Tampilkan frame utama
            if display_frames:
                try:
                    if len(display_frames) == 2:
                        display_frame = np.hstack((display_frames[0], display_frames[1]))
                        window_name = "Sistem Parkir (Masuk | Keluar)"
                    else:
                        display_frame = display_frames[0]
                        window_name = "Sistem Parkir (Masuk)" if frame_masuk is not None else "Sistem Parkir (Keluar)"
                    
                    cv2.imshow(window_name, display_frame)
                except Exception as e:
                    print(f"? Gagal menampilkan frame: {e}")
            
            # Reset OLED status jika diperlukan
            if oled_status_is_event and (time.time() - last_event_time > OLED_STANDBY_DELAY):
                update_oled("Sistem Siaga", "Mencari Plat...")
                oled_status_is_event = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n? Program dihentikan oleh pengguna")
    except Exception as e:
        print(f"? ERROR: {str(e)}")
    finally:
        # Bersihkan
        cam_masuk.stop()
        cam_keluar.stop()
        cam_masuk.join()
        cam_keluar.join()
        
        cv2.destroyAllWindows()
        
        if IS_HARDWARE_AVAILABLE:
            if chip:
                chip.close()
            if oled:
                update_oled("Sistem Off", "")
                time.sleep(1)
                oled.fill(0)
                oled.show()
        
        if sqlite_conn:
            sqlite_conn.close()
        
        print("? Program berhenti dengan bersih.")

if __name__ == "__main__":
    main()
