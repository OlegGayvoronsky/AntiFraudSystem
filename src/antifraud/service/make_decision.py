from antifraud.config import settings

MANUAL_REVIEW_THRESHOLD_A = getattr(settings, "manual_review_threshold", 0.2)


def decide(
    score: float,
    T_hight: float,
    T_low: float
) -> str:
    if score >= T_hight:
        return "BLOCK"
    if score < T_low:
        return "ALLOW"
    return "REVIEW"
