import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Khởi tạo mô hình (Tự động tải weights về máy ở lần chạy đầu tiên)
# "buffalo_l" là gói mô hình độ chính xác cao của InsightFace
#app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider']) 
#app.prepare(ctx_id=0, det_size=(640, 640))
ROOT_DIR = r"E:\mohinh\face_attendance"

# Khởi tạo mô hình và trỏ nó về ổ E của bạn
app = FaceAnalysis(
    name="buffalo_l", 
    root=ROOT_DIR, 
    providers=['CPUExecutionProvider'] #nếu máy có gpu thì ['CUDAExecutionProvider']
) 
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embeddings(image_path_or_frame):
    """
    Hàm nhận vào đường dẫn ảnh hoặc khung hình OpenCV, 
    trả về danh sách các khuôn mặt phát hiện được.
    """
    if isinstance(image_path_or_frame, str):
        img = cv2.imread(image_path_or_frame)
    else:
        img = image_path_or_frame

    if img is None:
        return []
        
    faces = app.get(img)
    return faces

def compute_similarity(embedding1, embedding2):
    """Tính toán khoảng cách Cosine giữa 2 vector khuôn mặt"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)