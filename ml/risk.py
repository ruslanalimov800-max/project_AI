"""Интерпретируемый риск-скоринг 0–100."""
def calculate_risk(p_sms=None, p_url=None, rule_groups=0):
    components = {}
    if p_sms is not None:
        components["sms"] = 55.0 * max(0.0, min(1.0, float(p_sms)))
    if p_url is not None:
        components["url"] = 35.0 * max(0.0, min(1.0, float(p_url)))
    components["rules"] = min(30.0, max(0, int(rule_groups)) * 5.0)
    score = min(100, round(sum(components.values())))
    level = "low" if score < 30 else "medium" if score < 60 else "high"
    return {"risk_score": score, "risk_level": level, "components": components}
