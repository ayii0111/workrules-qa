"""試算：特別休假天數、加班費。

這類「有明確公式」的問題交給程式計算，不交給 LLM：
LLM 擅長理解問題、解釋條文，但算數容易出錯；程式算的結果可以被測試、可以被驗證。
"""

from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction


# ── 特別休假（勞動基準法第 38 條）────────────────────────────

def months_between(start: date, end: date) -> int:
    """滿幾個月（未滿一個月不計）。

    結束月份沒有對應的日期時（例如 8/31 起算，2 月沒有 31 日），
    以該月最後一天為期滿日（參照民法第 121 條第 2 項）。
    """
    months = (end.year - start.year) * 12 + (end.month - start.month)
    is_month_end = (end + timedelta(days=1)).month != end.month
    if end.day < start.day and not is_month_end:
        months -= 1
    return max(months, 0)


def annual_leave_days(start: date, on: date) -> int:
    """依到職日計算，在 on 這天時「這一段年資」可以享有的特休天數（週年制）。"""
    months = months_between(start, on)
    years = months // 12
    if months < 6:
        return 0
    if years < 1:
        return 3
    if years < 2:
        return 7
    if years < 3:
        return 10
    if years < 5:
        return 14
    if years < 10:
        return 15
    return min(30, 15 + (years - 9))


def leave_schedule(start: date, years: int = 12) -> list[tuple[date, int]]:
    """列出每個特休起算日與當次天數：到職滿 6 個月一次，之後每滿一年一次。"""
    def add_months(d: date, n: int) -> date:
        y, m = divmod(d.month - 1 + n, 12)
        year, month = d.year + y, m + 1
        # 到職日是 31 號而該月沒有 31 號時，取該月最後一天
        for day in (d.day, 30, 29, 28):
            try:
                return date(year, month, day)
            except ValueError:
                continue
        raise ValueError(d)

    points = [add_months(start, 6)] + [add_months(start, 12 * y) for y in range(1, years + 1)]
    return [(p, annual_leave_days(start, p)) for p in points]


# ── 加班費（勞動基準法第 24 條）───────────────────────────────

WORKDAY_RATES = [(2, Fraction(4, 3)), (2, Fraction(5, 3))]
RESTDAY_RATES = [(2, Fraction(4, 3)), (6, Fraction(5, 3)), (4, Fraction(8, 3))]


@dataclass
class OvertimeLine:
    hours: float
    rate: Fraction
    amount: float


def hourly_wage(monthly_salary: float) -> float:
    """月薪制的平日每小時工資額：月薪 ÷ 30 ÷ 8。"""
    return monthly_salary / 30 / 8


def overtime_pay(monthly_salary: float, hours: float, day_type: str = "workday") -> list[OvertimeLine]:
    """平日加班：前 2 小時 ×4/3，第 3~4 小時 ×5/3。
    休息日加班：前 2 小時 ×4/3，第 3~8 小時 ×5/3，第 9~12 小時 ×8/3。"""
    tiers = WORKDAY_RATES if day_type == "workday" else RESTDAY_RATES
    limit = sum(h for h, _ in tiers)
    if hours > limit:
        raise ValueError(f"{'平日' if day_type == 'workday' else '休息日'}加班時數最多 {limit} 小時")
    wage, remaining, lines = hourly_wage(monthly_salary), hours, []
    for tier_hours, rate in tiers:
        h = min(remaining, tier_hours)
        if h <= 0:
            break
        lines.append(OvertimeLine(h, rate, round(wage * h * float(rate))))
        remaining -= h
    return lines
