from antifraud.config import settings

MANUAL_REVIEW_THRESHOLD_A = getattr(settings, "manual_review_threshold", 0.2)


def decide(score: float, threshold_b: float, threshold_a: float = MANUAL_REVIEW_THRESHOLD_A):
    if score >= threshold_b:
        return "ФРОД", "ЗАБЛОКИРОВАТЬ", "fraud"
    if score < threshold_a:
        return "НЕ ФРОД", "ПРОПУСТИТЬ", "not-fraud"
    return "НЕОПРЕДЕЛЕНО", "НА РУЧНУЮ ПРОВЕРКУ", "review"
