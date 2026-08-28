import datetime
import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import User, get_db


JWT_SECRET = os.environ.get(
    "JWT_SECRET",
    "doi-secret-nay-truoc-khi-deploy-that"
)

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto"
)

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(user_id: int) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + datetime.timedelta(days=JWT_EXPIRE_DAYS),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Token không hợp lệ."
            )

        return int(user_id)

    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Token không hợp lệ hoặc đã hết hạn."
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Chưa đăng nhập."
        )

    user_id = decode_token(credentials.credentials)

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Người dùng không tồn tại."
        )

    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):

    if credentials is None:
        return None

    try:
        user_id = decode_token(credentials.credentials)
    except HTTPException:
        return None

    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )
