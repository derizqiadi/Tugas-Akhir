import cv2
import math
import os
import time
from datetime import datetime
from ultralytics import YOLO

try:
    import gpiod
    import board
    import busio
    from PIL import Image, ImageDraw, ImageFont
    import adafruit_ssd1306
    IS_HARDWARE_AVAILABLE = True
    print("? Library hardware (gpiod, adafruit) ditemukan. Mode hardware diaktifkan.")
except ImportError:
    IS_HARDWARE_AVAILABLE = False
    print("??  Menjalankan dalam mode simulasi PC (tanpa kontrol hardware).")

# =============================================================================
# CLASS TRACKER 
# =============================================================================
class Tracker:
    """Tracker sederhana berbasis Centroid."""
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, id])
                    same_object_detected = True
                    break
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points.get(object_id)
            if center: new_center_points[object_id] = center
        self.center_points = new_center_points
        return objects_bbs_ids
    
# =============================================================================
# KONFIGURASI UTAMA
# =============================================================================
YOLO_MODEL_PATH = 'best.pt'
WEBCAM_INDEX = 0
SAVE_DIR = "hasil_crop"

CONF_THRESHOLD = 0.05
MIN_ASPECT_RATIO = 1.5
MAX_ASPECT_RATIO = 5.0
YOLO_IMG_SZ = 240
VIRTUAL_LINE_Y = 280
LINE_OFFSET = 15

# Konfigurasi Pin GPIO & Hardware
LED_PIN = 17
BUZZER_PIN = 27
GPIO_CHIP = 'gpiochip4'
OLED_WIDTH = 128
OLED_HEIGHT = 32
OLED_STANDBY_DELAY = 3 # Detik sebelum OLED kembali ke mode siaga

# =============================================================================
# INISIALISASI
# =============================================================================
print("?? Menginisialisasi sistem tes...")
model = YOLO(YOLO_MODEL_PATH)
tracker = Tracker()
processed_ids = []
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Folder '{SAVE_DIR}' berhasil dibuat.")

chip, led_line, buzzer_line, oled, font, font_kecil = None, None, None, None, None, None
if IS_HARDWARE_AVAILABLE:
    try:
        chip = gpiod.Chip(GPIO_CHIP)
        led_line, buzzer_line = chip.get_line(LED_PIN), chip.get_line(BUZZER_PIN)
        led_line.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)
        buzzer_line.request(consumer="buzzer", type=gpiod.LINE_REQ_DIR_OUT)
        i2c = board.I2C()
        oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=0x3c)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_kecil = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        print("? Perangkat keras berhasil diinisialisasi.")
    except Exception as e:
        print(f"? Gagal inisialisasi hardware: {e}. Lanjut mode simulasi.")
        IS_HARDWARE_AVAILABLE = False
print("? Inisialisasi selesai.")

# =============================================================================
# FUNGSI KONTROL HARDWARE
# =============================================================================
def update_oled(line1, line2=""):
    if not IS_HARDWARE_AVAILABLE: print(f"[OLED SIM] L1: {line1} | L2: {line2}"); return
    try:
        image = Image.new("1", (oled.width, oled.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
        draw.text((0, 0), line1, font=font, fill=255)
        draw.text((0, 16), line2, font=font_kecil, fill=255)
        oled.image(image); oled.show()
    except Exception as e: print(f"Error update OLED: {e}")

def signal_event_line_cross():
    """Sinyal untuk event plat melintasi garis: LED dan Buzzer menyala sekali."""
    if not IS_HARDWARE_AVAILABLE: print("[EVENT SIM] ? Beep & Blink Event!"); return
    led_line.set_value(1)
    buzzer_line.set_value(1)
    time.sleep(0.3) # Durasi sinyal lebih singkat
    led_line.set_value(0)
    buzzer_line.set_value(0)
    
# =============================================================================
# LOOP UTAMA
# =============================================================================
cap = cv2.VideoCapture(WEBCAM_INDEX)
oled_status_is_event = False
last_event_time = 0
update_oled("Sistem Siaga", "Mencari Plat...")

if not cap.isOpened():
    print(f"? ERROR: Tidak bisa membuka webcam dengan indeks {WEBCAM_INDEX}.")
    update_oled("ERROR", "Webcam Gagal")
else:
    print(f"? Webcam berhasil dibuka. Menjalankan deteksi...")
    print("Tekan 'q' pada jendela video untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret: print("?? Gagal membaca frame dari webcam."); break
        
        results = model.predict(frame, classes=[0], imgsz=YOLO_IMG_SZ, conf=CONF_THRESHOLD)
        detected_boxes = [list(map(int, box.xyxy[0])) for r in results for box in r.boxes]

        valid_plate_boxes = []
        for box in detected_boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if h > 0 and MIN_ASPECT_RATIO < (w/h) < MAX_ASPECT_RATIO:
                valid_plate_boxes.append(box)

        bbox_idx = tracker.update(valid_plate_boxes)
        
        for x1, y1, x2, y2, obj_id in bbox_idx:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cy = (y1 + y2) // 2
            
            if VIRTUAL_LINE_Y - LINE_OFFSET < cy < VIRTUAL_LINE_Y + LINE_OFFSET:
                if obj_id not in processed_ids:
                    processed_ids.append(obj_id)
                    print(f"\n? Plat ID:{obj_id} melintasi garis! Memicu hardware & preview...")
                    update_oled("PLAT MELINTAS!", f"ID Objek: {obj_id}")
                    signal_event_line_cross()
                    
                    oled_status_is_event = True
                    last_event_time = time.time()
                    
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0:
                        cv2.imshow("Preview Crop Plat Nomor", plate_crop)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"plate_{timestamp}.jpg"
                        save_path = os.path.join(SAVE_DIR, filename)
                        cv2.imwrite(save_path, plate_crop)
                        print(f"    -> Gambar disimpan: {save_path}")

        # ==> LOGIKA UNTUK MENGEMBALIKAN OLED KE MODE SIAGA <==
        if oled_status_is_event and (time.time() - last_event_time > OLED_STANDBY_DELAY):
            print("...OLED kembali ke mode siaga.")
            update_oled("Sistem Siaga", "Mencari Plat...")
            oled_status_is_event = False

        cv2.line(frame, (0, VIRTUAL_LINE_Y), (frame.shape[1], VIRTUAL_LINE_Y), (0, 0, 255), 2)
        cv2.imshow("Tes YOLO Webcam - Tekan 'q' untuk Keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Bersihkan setelah selesai
cap.release()
cv2.destroyAllWindows()
if IS_HARDWARE_AVAILABLE:
    if chip: chip.close()
    if oled: update_oled("Sistem Off", ""); time.sleep(1); oled.fill(0); oled.show()
print("Program tes selesai.")