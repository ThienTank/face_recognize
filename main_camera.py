import cv2
import pickle
import os
import numpy as np
import datetime
import time # Thêm thư viện time để làm bộ đếm thời gian chờ
from core_ai.face_utils import get_face_embeddings, compute_similarity
from core_ai.anti_spoofing import AntiSpoofing
from core_ai.database import log_attendance # <-- GỌI DATABASE VÀO ĐÂY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "models", "face_db.pkl")

# --- TẠO THƯ MỤC LƯU ẢNH MINH CHỨNG ---
RECORDS_DIR = os.path.join(BASE_DIR, "attendance_records")
os.makedirs(RECORDS_DIR, exist_ok=True)

# 1. Load Database Khuôn mặt
try:
    with open(DATABASE_PATH, 'rb') as f:
        face_db = pickle.load(f)
    print(f"Đã tải dữ liệu của {len(face_db)} người.")
except FileNotFoundError:
    print("CẢNH BÁO: Chưa có database! Hãy chạy file enroll.py trước.")
    face_db = {}

# 2. Khởi tạo mô hình Chống giả mạo
print("Đang khởi tạo Anti-Spoofing...")
anti_spoof_detector = AntiSpoofing()

def main():
    cap = cv2.VideoCapture(0)
    THRESHOLD = 0.60

    print("Đã bật Camera! Bấm phím 'q' trên cửa sổ camera để thoát.")
    
    # Bộ nhớ tạm lưu: { "Tên_Nhân_Viên": Thời_gian_ghi_nhận_cuối_cùng }
    # Giúp camera không ghi database liên tục 30 lần/giây cho cùng 1 người
    last_logged_time = {} 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera.")
            break
        
        faces = get_face_embeddings(frame)
        current_timestamp = time.time() # Lấy thời gian hiện tại tính bằng giây
        
        for face in faces:
            bbox = face.bbox.astype(int)
            
            # --- BƯỚC 1: KIỂM TRA CHỐNG GIẢ MẠO ---
            liveness_status, liveness_score = anti_spoof_detector.predict(frame, bbox)
            
            if liveness_status == "Fake":
                color = (0, 0, 255)
                label = f"SPOOF ({liveness_score*100:.1f}%)"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue 
                
            # --- BƯỚC 2: NHẬN DIỆN DANH TÍNH ---
            embedding = face.embedding
            best_match_name = "Unknown"
            best_score = -1
            
            for name, db_embedding in face_db.items():
                score = compute_similarity(embedding, db_embedding)
                if score > best_score:
                    best_score = score
                    best_match_name = name
                    
            if best_score >= THRESHOLD:
                # --- BƯỚC 3: KIỂM TRA COOLDOWN VÀ GHI DATABASE ---
                # Nếu người này chưa từng được lưu HOẶC đã đứng trước camera hơn 60 giây kể từ lần lưu trước
                if (best_match_name not in last_logged_time) or (current_timestamp - last_logged_time[best_match_name] > 60):
                    
                    now = datetime.datetime.now()
                    time_str = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"{best_match_name}_{time_str}.jpg"
                    filepath = os.path.join(RECORDS_DIR, filename)
                    
                    # 1. Chụp và lưu ảnh minh chứng gốc
                    cv2.imwrite(filepath, frame)
                    
                    # 2. Bắn dữ liệu vào SQL Database (Tự động tính Check-in / Check-out)
                    db_status = log_attendance(best_match_name, filepath)
                    print(f"[{best_match_name}] -> {db_status}")
                    
                    # 3. Cập nhật lại thời gian vừa ghi nhận để kích hoạt thời gian chờ
                    last_logged_time[best_match_name] = current_timestamp

                color = (0, 255, 0)
                label = f"{best_match_name} ({best_score:.2f})"
            else:
                color = (0, 165, 255)
                label = "Unknown"
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow("He Thong Diem Danh AI (Real-time)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
