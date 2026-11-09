"""Logic tính credit + quyền sử dụng cho INTELIGENT.

Tách riêng khỏi main.py để admin_api.py cũng dùng chung (không import vòng).
"""
import json
from datetime import date

from sqlalchemy.orm import Session

from database import UsageLog, User

# Credit theo gói — mỗi lượt chat tốn cố định, dài hay ngắn đều như nhau
# Credit làm mới mỗi ngày lúc 00:00 UTC (theo giờ server)
#   Free        : 100 credit, 12.5 credit / lượt chat  (~8 lượt)
#   Cold        : 500 credit, 20 credit / lượt chat    (~25 lượt)
#   Super Cold  : 1000 credit, 10 credit / lượt chat   (~100 lượt)
PLAN_CREDITS = {
    "free":                  {"total": 100,  "cost": 12.5},
    "inteligent_cold":       {"total": 500,  "cost": 20},
    "inteligent_super_cold": {"total": 1000, "cost": 10},
}
CREDIT_FEATURE = "chat"

PLAN_LABELS = {
    "free": "Free",
    "inteligent_cold": "INTELIGENT Cold",
    "inteligent_super_cold": "INTELIGENT Super Cold",
}


def _fmt(n) -> str:
    return f"{n:g}"


def get_user_plan(user) -> str:
    """Gói đang dùng: ưu tiên gói user tự chọn (lưu trong settings), rồi tới cột plan, mặc định free."""
    try:
        chosen = json.loads(getattr(user, "settings_json", None) or "{}").get("plan")
    except Exception:
        chosen = None
    for plan in (chosen, getattr(user, "plan", None)):
        if plan in PLAN_CREDITS:
            return plan
    return "free"


def set_user_plan(user, plan: str):
    """Đặt gói (admin dùng) -> ghi cả cột plan lẫn settings để chắc chắn có hiệu lực."""
    if plan not in PLAN_CREDITS:
        return
    user.plan = plan
    try:
        settings = json.loads(user.settings_json or "{}")
    except Exception:
        settings = {}
    settings["plan"] = plan
    user.settings_json = json.dumps(settings)


def is_blocked(user) -> bool:
    return bool(getattr(user, "is_blocked", False))


def _today_log(db: Session, user):
    return (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id, UsageLog.feature == CREDIT_FEATURE,
                UsageLog.usage_date == date.today().isoformat())
        .first()
    )


def credit_status(db: Session, user) -> dict:
    """Trạng thái credit hôm nay của user.

    Hạn mức = (hạn mức riêng của admin nếu có) hoặc (credit gói + credit admin cộng thêm).
    """
    plan = get_user_plan(user)
    cfg = PLAN_CREDITS[plan]
    log = _today_log(db, user)
    used_chats = log.count if log else 0

    base_total = cfg["total"]
    extra = float(getattr(user, "extra_credits", 0) or 0)
    custom = getattr(user, "custom_limit", None)

    if custom is not None and int(custom) > 0:
        total = float(custom)
        limited_by = "custom"
    else:
        total = base_total + extra
        limited_by = "plan"

    cost = cfg["cost"]
    remaining = max(total - used_chats * cost, 0)
    return {
        "plan": plan,
        "plan_label": PLAN_LABELS.get(plan, plan),
        "total": total,
        "base_total": base_total,
        "extra_credits": extra,
        "custom_limit": custom,
        "limited_by": limited_by,
        "cost": cost,
        "used_chats": used_chats,
        "remaining": remaining,
        "max_chats": int(total // cost),
        "remaining_chats": int(remaining // cost),
        "is_blocked": is_blocked(user),
    }


def consume_chat_credit(db: Session, user):
    """Trừ credit cho 1 lượt chat. Trả về (lỗi hoặc None, log để hoàn lại nếu chat thất bại).
    Khách chưa đăng nhập (user=None) thì không tính credit."""
    if not user:
        return None, None
    if is_blocked(user):
        return ("Tài khoản của bạn đang bị khoá. Liên hệ quản trị viên nếu cần hỗ trợ.", None)
    st = credit_status(db, user)
    if st["remaining"] < st["cost"]:
        return (
            f"Bạn đã dùng hết {_fmt(st['total'])} credit hôm nay (mỗi lượt chat tốn {_fmt(st['cost'])} credit). "
            f"Credit làm mới lúc 00:00 UTC, hoặc đổi sang gói khác trong Cài đặt.",
            None,
        )
    log = _today_log(db, user)
    if log:
        log.count += 1
    else:
        log = UsageLog(user_id=user.id, feature=CREDIT_FEATURE, usage_date=date.today().isoformat(), count=1)
        db.add(log)
    db.commit()
    return None, log


def refund_chat_credit(db: Session, log):
    """Hoàn credit nếu model lỗi (người dùng không bị trừ oan)."""
    if log is not None and log.count > 0:
        log.count -= 1
        db.commit()


def reset_today_usage(db: Session, user) -> int:
    """Xoá lượt dùng hôm nay của 1 user -> trả về số lượt đã xoá."""
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id, UsageLog.feature == CREDIT_FEATURE,
                UsageLog.usage_date == date.today().isoformat())
        .all()
    )
    n = sum(l.count for l in logs)
    for l in logs:
        db.delete(l)
    db.commit()
    return n
