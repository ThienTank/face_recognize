import os
import cv2
import numpy as np
import onnxruntime as ort

# Đường dẫn tự động tìm file anti_spoof.onnx trong thư mục models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "anti_spoofing.onnx") 

class AntiSpoofing:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
        # Khởi tạo ONNX session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, frame, bbox):
        """
        Dự đoán Real/Fake.
        Trả về: (Trạng thái "Real" hoặc "Fake", Điểm tự tin)
        """
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Mở rộng bbox ra khoảng 20% để lấy bối cảnh viền điện thoại (nếu có)
        cw, ch = x2 - x1, y2 - y1
        x1 = max(0, x1 - int(cw * 0.2))
        y1 = max(0, y1 - int(ch * 0.2))
        x2 = min(w, x2 + int(cw * 0.2))
        y2 = min(h, y2 + int(ch * 0.2))
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return "Fake", 0.0
            
        # Chuẩn bị ảnh cho MiniFASNet: kích thước 80x80
        face_img = cv2.resize(face_img, (80, 80))
        face_img = face_img.astype(np.float32)
        face_img = np.transpose(face_img, (2, 0, 1)) # Chuyển từ HWC sang CHW
        face_img = np.expand_dims(face_img, axis=0)  # Thêm chiều batch (1, C, H, W)
        
        # Chạy mô hình
        outputs = self.session.run(None, {self.input_name: face_img})
        result = outputs[0][0]
        
        # Tính xác suất bằng hàm Softmax
        exp_result = np.exp(result - np.max(result))
        probabilities = exp_result / np.sum(exp_result)
        
        # Index 1 là nhãn Real (Thật), Index 0 là Fake (Giả)
        real_score = probabilities[1] 
        
        if real_score > 0.6:  # Ngưỡng tin cậy, có thể chỉnh sửa
            return "Real", real_score
        else:
            return "Fake", (1 - real_score)