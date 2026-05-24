import os
import cv2
import pickle
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core_ai.face_utils import get_face_embeddings, compute_similarity
from core_ai.anti_spoofing import AntiSpoofing  # <-- Import Module mới

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "models", "face_db.pkl")

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Khởi tạo mô hình Chống giả mạo
print("Đang khởi tạo Anti-Spoofing...")
anti_spoof_detector = AntiSpoofing()

try:
    with open(DATABASE_PATH, 'rb') as f:
        face_db = pickle.load(f)
except FileNotFoundError:
    face_db = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Không thể đọc ảnh"}

    faces = get_face_embeddings(img)
    
    recognized_list = []
    unknown_count = 0
    fake_count = 0  # Đếm số mặt giả mạo
    THRESHOLD = 0.45
    
    for face in faces:
        bbox = face.bbox.astype(int)
        
        # 1. KIỂM TRA CHỐNG GIẢ MẠO TRƯỚC
        liveness_status, liveness_score = anti_spoof_detector.predict(img, bbox)
        
        if liveness_status == "Fake":
            fake_count += 1
            color = (0, 0, 255) # Đỏ cảnh báo
            label = f"SPOOF ({liveness_score*100:.1f}%)"
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            continue # Dừng luôn, bỏ qua bước nhận diện khuôn mặt
            
        # 2. NẾU LÀ MẶT THẬT THÌ MỚI NHẬN DIỆN DANH TÍNH
        embedding = face.embedding
        best_match_name = "Unknown"
        best_score = -1
        
        for name, db_embedding in face_db.items():
            score = compute_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match_name = name
                
        if best_score >= THRESHOLD:
            recognized_list.append(best_match_name)
            color = (0, 255, 0) # Xanh lá
            label = best_match_name
        else:
            unknown_count += 1
            color = (0, 165, 255) # Cam cho người lạ
            label = "Unknown"
            
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    recognized_list = list(set(recognized_list))
    
    return {
        "total_faces": len(faces),
        "recognized_count": len(recognized_list),
        "recognized_list": recognized_list,
        "unknown_count": unknown_count,
        "fake_count": fake_count, # Trả về web số lượng mặt giả
        "image_base64": img_base64
    }