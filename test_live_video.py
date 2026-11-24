from datetime import datetime
import os
import subprocess
import time
import cv2
import numpy as np
import requests
import urllib3
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration comes from environment variables (see .env.example).
# Nothing about the stream is hard-coded.
#
# NOTE: Before using any IZUM / Izmir Metropolitan Municipality camera
# stream, check IZUM's terms of use. Use only streams you are permitted to
# access, and do not disguise this client as a web browser.
# ---------------------------------------------------------------------------
def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


url = os.environ.get("IZUM_STREAM_URL", "").strip()
if not url:
    raise SystemExit(
        "IZUM_STREAM_URL is not set. Copy .env.example to .env, fill in a "
        "stream URL you are permitted to use, and export it "
        "(e.g. `set -a; source .env; set +a`)."
    )

VERIFY_TLS = _env_bool("IZUM_STREAM_VERIFY_TLS", True)
if not VERIFY_TLS:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Optional request headers; empty by default (no browser impersonation).
headers = {}
if os.environ.get("IZUM_STREAM_USER_AGENT", "").strip():
    headers["User-Agent"] = os.environ["IZUM_STREAM_USER_AGENT"].strip()
if os.environ.get("IZUM_STREAM_REFERER", "").strip():
    headers["Referer"] = os.environ["IZUM_STREAM_REFERER"].strip()

# Modeli yükle
model = YOLO(os.environ.get("EDGE_YOLO_WEIGHTS", "yolov8n.pt"))

CLASS_MAPPING = {
    2: "Otomobil",
    3: "Motosiklet",
    5: "Otobüs",
    7: "Kamyon"
}
TARGET_CLASSES = list(CLASS_MAPPING.keys())

def log(msg, level="INFO"):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{level}] {msg}")

def bring_window_to_front():
    """macOS üzerinde pencereyi doğrudan öne getirir."""
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Python" to activate'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

def draw_info_panel(frame, counts):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (240, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "CANLI ARAC SAYIMI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset = 65
    for cls_id, name in CLASS_MAPPING.items():
        count = counts.get(cls_id, 0)
        text = f"{name}: {count}"
        cv2.putText(frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_offset += 24

def show_reconnect_screen(window_name, msg="Baglanti koptu. Yeniden baglaniliyor..."):
    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, msg, (200, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow(window_name, placeholder)
    cv2.waitKey(1)

window_name = "IZUM Trafik Analizi"

# Serbest boyutlandırma desteği
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

window_initialized = False
retry_count = 0
max_reconnect_delay = 5

while True:
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        log("Pencere kapatildi. Program sonlandiriliyor.", "INFO")
        break

    try:
        log(f"Akisa baglaniliyor: {url}", "INFO")
        response = requests.get(
            url, headers=headers, stream=True, verify=VERIFY_TLS, timeout=(5, 10)
        )

        if response.status_code != 200:
            log(f"Sunucu hatasi alindi (HTTP {response.status_code})", "WARN")
            show_reconnect_screen(window_name, f"HTTP Hatasi: {response.status_code} Bekleniyor...")
            time.sleep(2)
            continue

        log("Baglanti basarili, veri akisi baslatildi.", "INFO")
        retry_count = 0
        byte_data = b""

        for chunk in response.iter_content(chunk_size=4096):
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if not chunk:
                log("Veri akisi beklenmedik sekilde kesildi (Bos paket).", "WARN")
                break

            byte_data += chunk
            a = byte_data.find(b"\xff\xd8")
            b = byte_data.find(b"\xff\xd9")

            if a != -1 and b != -1:
                jpg = byte_data[a : b + 2]
                byte_data = byte_data[b + 2 :]

                try:
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                except Exception as decode_err:
                    log(f"Kare cozumleme hatasi: {decode_err}", "DEBUG")
                    continue

                if frame is not None:
                    # Başlangıç boyutunu 1440px genişliğe büyüt ve öne çek
                    if not window_initialized:
                        h, w = frame.shape[:2]
                        target_width = 1440
                        target_height = int(target_width * (h / w))
                        cv2.resizeWindow(window_name, target_width, target_height)
                        bring_window_to_front()
                        window_initialized = True

                    try:
                        results = model.predict(
                            frame,
                            classes=TARGET_CLASSES,
                            conf=0.35,
                            device="mps",
                            verbose=False,
                        )
                        annotated_frame = results[0].plot()

                        counts = {cls_id: 0 for cls_id in TARGET_CLASSES}
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            classes_detected = results[0].boxes.cls.int().tolist()
                            for c in classes_detected:
                                if c in counts:
                                    counts[c] += 1

                        draw_info_panel(annotated_frame, counts)
                        cv2.imshow(window_name, annotated_frame)

                    except Exception as model_err:
                        log(f"Model cikarim hatasi: {model_err}", "ERROR")
                        cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log("Kullanici 'q' tusuna basti. Cikiliyor.", "INFO")
                    cv2.destroyAllWindows()
                    exit(0)

    except requests.exceptions.Timeout:
        log("Baglanti zaman asimina ugradi (Timeout).", "WARN")
        show_reconnect_screen(window_name, "Zaman Asimi - Yeniden Baglaniliyor...")
    except requests.exceptions.SSLError as ssl_err:
        log(f"SSL Baglanti Hatasi: {ssl_err}", "ERROR")
        show_reconnect_screen(window_name, "SSL Hatasi - Tekrar Deneniyor...")
    except requests.exceptions.ConnectionError:
        log("Ag veya sunucu erisim hatasi (ConnectionError).", "WARN")
        show_reconnect_screen(window_name, "Baglanti Kesildi - Bekleniyor...")
    except Exception as e:
        log(f"Beklenmeyen bir hata olustu: {e}", "CRITICAL")
        show_reconnect_screen(window_name, "Beklenmeyen Hata Olustu!")

    retry_count += 1
    sleep_time = min(retry_count * 1.5, max_reconnect_delay)
    log(f"{sleep_time:.1f} saniye sonra tekrar baglanilacak (Deneme: {retry_count})...", "INFO")
    time.sleep(sleep_time)

cv2.destroyAllWindows()