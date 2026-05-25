import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CẤU HÌNH DATABASE ---
# Nếu đã cài MySQL/XAMPP, hãy bỏ comment dòng dưới và sửa lại user/password:
# DB_URL = "mysql+pymysql://root:123456@localhost/face_attendance"

# Tạm thời để SQLite mặc định để chạy được ngay lập tức mà không cần cài server
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, 'attendance.db')}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ĐỊNH NGHĨA BẢNG LƯU TRỮ ---
class Attendance(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(20), default="NV_UNKNOWN") # Tạm thời gán mặc định
    name = Column(String(50), index=True)
    date = Column(Date, index=True)
    check_in = Column(DateTime, nullable=True)
    check_out = Column(DateTime, nullable=True)
    image_path = Column(String(255), nullable=True)

# Lệnh này sẽ tự động tạo file .db (hoặc tạo bảng trong MySQL) nếu chưa có
Base.metadata.create_all(bind=engine)

# --- HÀM XỬ LÝ CHECK-IN / CHECK-OUT ---
def log_attendance(name, image_path):
    db = SessionLocal()
    today = datetime.date.today()
    now = datetime.datetime.now()
    
    # Tìm xem hôm nay người này đã điểm danh lần nào chưa
    record = db.query(Attendance).filter(Attendance.name == name, Attendance.date == today).first()
    
    if not record:
        # CHƯA CÓ -> Lưu CHECK-IN
        new_record = Attendance(
            name=name,
            employee_id=f"NV_{name.upper()}", # Giả lập mã NV từ tên
            date=today,
            check_in=now,
            image_path=image_path
        )
        db.add(new_record)
        db.commit()
        status = f"✅ Check-in thành công ({now.strftime('%H:%M:%S')})"
    else:
        # ĐÃ CÓ -> Lưu CHECK-OUT
        # Mẹo: Cần cách lần Check-in ít nhất 10 phút (600 giây) thì mới cho Check-out, 
        # để tránh việc đứng trước camera vài giây bị tính luôn là check-out.
        time_diff = (now - record.check_in).total_seconds()
        
        if time_diff > 600:
            record.check_out = now
            db.commit()
            status = f"👋 Check-out thành công ({now.strftime('%H:%M:%S')})"
        else:
            status = f"⏳ Đã check-in gần đây (Vui lòng đợi 10p để check-out)"
            
    db.close()
    return status