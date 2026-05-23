import os
import pickle
import numpy as np
from face_utils import get_face_embeddings

# KNOWN_FACES_DIR = "../data/known_faces"
# DATABASE_PATH = "../models/face_db.pkl"
# Tự động lấy đường dẫn tuyệt đối của thư mục gốc dự án (face_attendance)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")
DATABASE_PATH = os.path.join(BASE_DIR, "models", "face_db.pkl")

# Tự động tạo thư mục nếu bạn lỡ quên chưa tạo
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

def enroll_faces():
    face_db = {}
    print("Đang bắt đầu quá trình trích xuất đặc trưng (Enrollment)...")

    # Lặp qua từng thư mục người dùng
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        embeddings_list = []
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            faces = get_face_embeddings(image_path)
            
            if len(faces) == 1:
                # Lấy vector đặc trưng (embedding) của khuôn mặt
                embeddings_list.append(faces[0].embedding)
            elif len(faces) > 1:
                print(f"Bỏ qua ảnh {image_path}: Có nhiều hơn 1 khuôn mặt.")
            else:
                print(f"Bỏ qua ảnh {image_path}: Không tìm thấy khuôn mặt.")
        
        if embeddings_list:
            # Lấy trung bình cộng các vector của cùng một người để độ chính xác cao hơn
            avg_embedding = np.mean(embeddings_list, axis=0)
            # Chuẩn hóa vector
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            face_db[person_name] = avg_embedding
            print(f"Đã lấy mẫu thành công cho: {person_name}")

    # Lưu Database ra file
    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(face_db, f)
    print(f"Đã lưu cơ sở dữ liệu tại {DATABASE_PATH}")

if __name__ == "__main__":
    enroll_faces()