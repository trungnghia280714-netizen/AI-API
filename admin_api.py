
import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from auth import get_admin_user
from credits import (
    PLAN_CREDITS,
    credit_status,
    reset_today_usage,
    set_user_plan,
)
from database import Conversation, Message, UsageLog, User, get_db

router = APIRouter(prefix="/api/admin", tags=["admin"])

UTC_TODAY = "chat"


def _day_start():
    return datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _user_row(db: Session, u: User) -> dict:
    st = credit_status(db, u)
    conv_count = db.query(func.count(Conversation.id)).filter(Conversation.user_id == u.id).scalar() or 0
    msg_count = db.query(func.count(Message.id)).join(
        Conversation, Message.conversation_id == Conversation.id
    ).filter(Conversation.user_id == u.id).scalar() or 0
    chats_today = (
        db.query(func.coalesce(func.sum(UsageLog.count), 0))
        .filter(UsageLog.user_id == u.id, UsageLog.feature == UTC_TODAY,
                UsageLog.usage_date == datetime.date.today().isoformat())
        .scalar()
        or 0
    )
    return {
        "id": u.id,
        "email": u.email,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "plan": st["plan"],
        "plan_label": st["plan_label"],
        "is_admin": bool(u.is_admin),
        "is_blocked": bool(u.is_blocked),
        "extra_credits": float(u.extra_credits or 0),
        "custom_limit": u.custom_limit,
        "note": u.note or "",
        "has_password": bool(u.password_hash),
        "conversations": conv_count,
        "messages": msg_count,
        "chats_today": int(chats_today),
        "credits": st,
    }


@router.get("/me")
async def admin_me(user: User = Depends(get_admin_user)):
    return {"email": user.email, "id": user.id, "is_admin": True}


@router.get("/stats")
async def stats(user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    now = datetime.datetime.utcnow()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week = day - datetime.timedelta(days=7)

    total_users = db.query(func.count(User.id)).scalar() or 0
    admins = db.query(func.count(User.id)).filter(User.is_admin.is_(True)).scalar() or 0
    blocked = db.query(func.count(User.id)).filter(User.is_blocked.is_(True)).scalar() or 0
    new_today = db.query(func.count(User.id)).filter(User.created_at >= day).scalar() or 0
    new_week = db.query(func.count(User.id)).filter(User.created_at >= week).scalar() or 0

    total_convs = db.query(func.count(Conversation.id)).scalar() or 0
    total_msgs = db.query(func.count(Message.id)).scalar() or 0
    msgs_today = db.query(func.count(Message.id)).filter(Message.created_at >= day).scalar() or 0
    chats_today = (
        db.query(func.coalesce(func.sum(UsageLog.count), 0))
        .filter(UsageLog.feature == UTC_TODAY, UsageLog.usage_date == datetime.date.today().isoformat())
        .scalar()
        or 0
    )

    # top 5 dùng nhiều nhất hôm nay
    rows = (
        db.query(UsageLog.user_id, func.sum(UsageLog.count).label("c"))
        .filter(UsageLog.feature == UTC_TODAY, UsageLog.usage_date == datetime.date.today().isoformat())
        .group_by(UsageLog.user_id)
        .order_by(func.sum(UsageLog.count).desc())
        .limit(5)
        .all()
    )
    top = []
    for uid, c in rows:
        u = db.query(User).filter(User.id == uid).first()
        if u:
            top.append({"email": u.email, "chats_today": int(c or 0)})

    # tin nhắn mới nhất
    recent = (
        db.query(Message, User.email)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .join(User, Conversation.user_id == User.id)
        .order_by(Message.id.desc())
        .limit(10)
        .all()
    )
    recent_msgs = [
        {
            "id": m.id,
            "email": email,
            "role": m.role,
            "content": (m.content or "")[:300],
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m, email in recent
    ]

    return {
        "total_users": total_users,
        "admins": admins,
        "blocked": blocked,
        "new_today": new_today,
        "new_week": new_week,
        "total_conversations": total_convs,
        "total_messages": total_msgs,
        "messages_today": msgs_today,
        "chats_today": int(chats_today),
        "top_users_today": top,
        "recent_messages": recent_msgs,
        "plans": [{"id": k, "label": v} for k, v in PLAN_CREDITS.items()],
    }


@router.get("/users")
async def list_users(
    request: Request,
    q: str = "",
    plan: str = "",
    status: str = "",
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if q:
        like = f"%{q.strip().lower()}%"
        query = query.filter(func.lower(User.email).like(like))
    if plan:
        query = query.filter(or_(User.plan == plan,
                                 func.coalesce(User.settings_json, "").like(f'%"{plan}"%')))
    if status == "blocked":
        query = query.filter(User.is_blocked.is_(True))
    elif status == "active":
        query = query.filter(User.is_blocked.is_(False))
    elif status == "admin":
        query = query.filter(User.is_admin.is_(True))

    total = query.count()
    users = query.order_by(User.id.desc()).limit(min(limit, 200)).offset(max(offset, 0)).all()
    return {
        "total": total,
        "users": [_user_row(db, u) for u in users],
    }


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: int,
    user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản.")
    data = _user_row(db, u)

    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == u.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    data["conversation_list"] = [
        {
            "id": c.id,
            "title": c.title,
            "tab": c.tab,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "messages": len(c.messages),
        }
        for c in convs
    ]

    # lượt dùng 7 ngày gần nhất
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == u.id)
        .order_by(UsageLog.usage_date.desc())
        .limit(10)
        .all()
    )
    data["usage_logs"] = [
        {"date": l.usage_date, "feature": l.feature, "count": l.count} for l in logs
    ]
    return data


@router.patch("/users/{user_id}")
async def update_user(
    user_id: int,
    request: Request,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        return JSONResponse({"error": "Không tìm thấy tài khoản."}, status_code=404)

    if "plan" in body and body["plan"]:
        if body["plan"] not in PLAN_CREDITS:
            return JSONResponse({"error": "Gói không hợp lệ."}, status_code=400)
        set_user_plan(u, body["plan"])

    if "is_blocked" in body:
        u.is_blocked = bool(body["is_blocked"])

    if "is_admin" in body:
        if u.id == admin.id and not body["is_admin"]:
            return JSONResponse({"error": "Không thể tự bỏ quyền admin của chính mình."}, status_code=400)
        u.is_admin = bool(body["is_admin"])

    if "extra_credits" in body:
        try:
            u.extra_credits = max(float(body["extra_credits"] or 0), 0)
        except (TypeError, ValueError):
            return JSONResponse({"error": "extra_credits phải là số."}, status_code=400)

    if "custom_limit" in body:
        v = body["custom_limit"]
        if v in (None, "", 0, "0"):
            u.custom_limit = None
        else:
            try:
                u.custom_limit = max(int(v), 0)
            except (TypeError, ValueError):
                return JSONResponse({"error": "custom_limit phải là số nguyên."}, status_code=400)

    if "note" in body:
        u.note = str(body["note"] or "")[:2000]

    db.add(u)
    db.commit()
    db.refresh(u)
    return _user_row(db, u)


@router.post("/users/{user_id}/reset-usage")
async def reset_usage(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        return JSONResponse({"error": "Không tìm thấy tài khoản."}, status_code=404)
    n = reset_today_usage(db, u)
    return {"ok": True, "reset_chats": n, "user": _user_row(db, u)}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        return JSONResponse({"error": "Không tìm thấy tài khoản."}, status_code=404)
    if u.id == admin.id:
        return JSONResponse({"error": "Không thể xoá chính tài khoản đang đăng nhập."}, status_code=400)
    email = u.email
    db.delete(u)  # cascade xoá luôn conversations + messages
    db.commit()
    return {"ok": True, "deleted_email": email}


@router.get("/users/{user_id}/messages")
async def user_messages(
    user_id: int,
    limit: int = 100,
    offset: int = 0,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản.")
    rows = (
        db.query(Message, Conversation.title, Conversation.id)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == u.id)
        .order_by(Message.id.desc())
        .limit(min(limit, 500))
        .offset(max(offset, 0))
        .all()
    )
    return {
        "email": u.email,
        "messages": [
            {
                "id": m.id,
                "conversation_id": cid,
                "conversation_title": title,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m, title, cid in rows
        ],
    }


@router.get("/messages")
async def search_messages(
    q: str = "",
    limit: int = 100,
    offset: int = 0,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Tìm kiếm nội dung tin nhắn của mọi người dùng."""
    query = db.query(Message, User.email, Conversation.title, Conversation.id).join(
        Conversation, Message.conversation_id == Conversation.id
    ).join(User, Conversation.user_id == User.id)
    if q:
        query = query.filter(Message.content.ilike(f"%{q.strip()}%"))
    total = query.count()
    rows = query.order_by(Message.id.desc()).limit(min(limit, 300)).offset(max(offset, 0)).all()
    return {
        "total": total,
        "messages": [
            {
                "id": m.id,
                "email": email,
                "conversation_id": cid,
                "conversation_title": title,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m, email, title, cid in rows
        ],
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    c = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")
    u = db.query(User).filter(User.id == c.user_id).first()
    return {
        "id": c.id,
        "title": c.title,
        "tab": c.tab,
        "email": u.email if u else None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in c.messages
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    c = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not c:
        return JSONResponse({"error": "Không tìm thấy cuộc trò chuyện."}, status_code=404)
    db.delete(c)
    db.commit()
    return {"ok": True}
