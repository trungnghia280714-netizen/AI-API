import datetime
import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import User, get_db

# Đổi JWT_SECRET này trên Render (Environment) trước khi dùng thật, đừng để mặc định
JWT_SECRET = os.environ.get("JWT_SECRET", "doi-secret-nay-truoc-khi-deploy-that")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

# pbkdf2_sha256: thuần Python, không cần biên dịch C extension như bcrypt -> cài đặt ổn định hơn
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn.")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Bắt buộc phải đăng nhập, ném lỗi 401 nếu không có/token sai."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    user_id = decode_token(credentials.credentials)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Người dùng không tồn tại.")
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """Không bắt buộc đăng nhập -> trả None nếu chưa đăng nhập, dùng cho các route vẫn
    hoạt động ở chế độ khách nhưng có thể lưu lịch sử nếu người dùng đã đăng nhập."""
    if credentials is None:
        return None
    try:
        user_id = decode_token(credentials.credentials)
    except HTTPException:
        return None
    return db.query(User).filter(User.id == user_id).first()
