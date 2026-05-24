import cv2
import pickle
import os
import numpy as np
from core_ai.face_utils import get_face_embeddings, compute_similarity
from core_ai.anti_spoofing import AntiSpoofing  # <-- Import mô hình chống giả mạo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "models", "face_db.pkl")

# 1. Load Database Khuôn mặt
try:
    with open(DATABASE_PATH, 'rb') as f:
        face_db = pickle.load(f)
    print(f"Đã tải dữ liệu của {len(face_db)} người.")
except FileNotFoundError:
    print("CẢNH BÁO: Chưa có database! Hãy chạy file enroll.py trước.")
    face_db = {}

# 2. Khởi tạo mô hình Chống giả mạo (MiniFASNet)
print("Đang khởi tạo Anti-Spoofing...")
anti_spoof_detector = AntiSpoofing()

def main():
    # Mở camera (0 là camera mặc định của máy tính/laptop)
    cap = cv2.VideoCapture(0)
    THRESHOLD = 0.45

    print("Đã bật Camera! Bấm phím 'q' trên cửa sổ camera để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera.")
            break
        
        # 3. Quét tìm tất cả khuôn mặt trong khung hình
        faces = get_face_embeddings(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            
            # --- BƯỚC 1: KIỂM TRA CHỐNG GIẢ MẠO ---
            liveness_status, liveness_score = anti_spoof_detector.predict(frame, bbox)
            
            if liveness_status == "Fake":
                color = (0, 0, 255) # Đỏ cảnh báo
                label = f"SPOOF ({liveness_score*100:.1f}%)"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue # Dừng lại, bỏ qua người này
                
            # --- BƯỚC 2: NHẬN DIỆN DANH TÍNH (chỉ chạy khi là mặt thật) ---
            embedding = face.embedding
            best_match_name = "Unknown"
            best_score = -1
            
            for name, db_embedding in face_db.items():
                score = compute_similarity(embedding, db_embedding)
                if score > best_score:
                    best_score = score
                    best_match_name = name
                    
            if best_score >= THRESHOLD:
                color = (0, 255, 0) # Xanh lá
                label = f"{best_match_name} ({best_score:.2f})"
            else:
                color = (0, 165, 255) # Cam cho người lạ
                label = "Unknown"
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 4. Hiển thị khung hình lên màn hình
        cv2.imshow("He Thong Diem Danh AI (Real-time)", frame)
        
        # Lắng nghe phím 'q' để thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Dọn dẹp tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()