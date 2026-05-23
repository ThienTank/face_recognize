import cv2
import pickle
import time
from core_ai.face_utils import get_face_embeddings, compute_similarity

DATABASE_PATH = "models/face_db.pkl"
THRESHOLD = 0.45  # Ngưỡng chấp nhận (Có thể chỉnh từ 0.4 -> 0.6)

# Load Database đã lấy mẫu
try:
    with open(DATABASE_PATH, 'rb') as f:
        face_db = pickle.load(f)
    print("Đã tải xong cơ sở dữ liệu nhận dạng.")
except FileNotFoundError:
    print("Không tìm thấy file Database! Hãy chạy enroll.py trước.")
    exit()

# Khởi động Camera
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Tối ưu: Chỉ nhận diện mỗi 3 frame 1 lần để tránh giật lag (Frame skipping)
    if frame_count % 3 != 0:
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Lấy khuôn mặt từ khung hình hiện tại
    faces = get_face_embeddings(frame)

    for face in faces:
        bbox = face.bbox.astype(int) # Khung chữ nhật
        embedding = face.embedding
        
        best_match_name = "Unknown"
        best_score = -1
        
        # So sánh khuôn mặt trong camera với từng người trong Database
        for name, db_embedding in face_db.items():
            score = compute_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match_name = name

        # Logic nhận diện đúng hoặc sai
        if best_score >= THRESHOLD:
            color = (0, 255, 0) # Xanh lá cho người quen
            label = f"{best_match_name} ({best_score:.2f})"
        else:
            color = (0, 0, 255) # Đỏ cho người lạ
            label = f"Unknown ({best_score:.2f})"

        # Vẽ Bounding Box và Tên lên màn hình
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Attendance System", frame)
    
    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()