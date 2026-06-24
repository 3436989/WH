from datetime import datetime
import re

def get_current_week():
    """获取当前校历周数（假设2026年3月2日开学）"""
    today = datetime.now()
    start_date = datetime(2026, 3, 2)   # 固定开学日期
    delta = today - start_date
    week_num = delta.days // 7 + 1
    if week_num < 1:
        week_num = 1
    return f"📅 现在是第 {week_num} 周（校历）"

def calculate_gpa(scores_str):
    """计算绩点，输入格式：'85,90,78' """
    try:
        scores = [float(x.strip()) for x in scores_str.split(',') if x.strip()]
        total = 0.0
        for s in scores:
            if s >= 90:
                total += 4.0
            elif s >= 80:
                total += 3.0
            elif s >= 70:
                total += 2.0
            elif s >= 60:
                total += 1.0
            else:
                total += 0.0
        gpa = total / len(scores) if scores else 0.0
        return f"您的平均绩点（加权）为：{gpa:.2f}"
    except Exception as e:
        return f"输入格式有误，请使用逗号分隔的数字，如:85,90,78"