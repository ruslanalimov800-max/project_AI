"""Задание 13: Разработка алгоритма оценки риска (Risk Score 0–100).
Все веса, пороги и правила усиления загружаются из risk_config.yaml.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import yaml

from ml.rules import GROUP_TITLES
from ml.url_features import explain_url

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "risk_config.yaml"

DEFAULT_CONFIG = {
    "weights": {"w1_sms": 0.5, "w2_url": 0.3, "w3_se": 0.2},
    "se_normalization": {"max_groups_for_full_score": 3},
    "thresholds": {"medium_min": 35, "high_min": 70, "url_dangerous_t2": 0.55},
    "overrides": {
        "code_request_min_score": 70,
        "high_sms_and_dangerous_url_min_score": 85,
        "high_sms_threshold": 0.80,
        "ip_or_punycode_or_brand_min_score": 60,
        "multi_se_groups_count": 2,
        "multi_se_min_score": 72,
    },
    "recommendations": {
        "LOW": "Признаки мошенничества не обнаружены или незначительны. Сохраняйте бдительность и проверяйте отправителя.",
        "MEDIUM": "Обнаружены отдельные признаки риска. Не выполняйте срочные требования из сообщения и проверьте адрес сайта вручную.",
        "HIGH": "Не переходите по ссылке, не сообщайте коды и данные карты. Свяжитесь с банком по номеру с обратной стороны карты.",
    },
    "disclaimer": "Оценка носит вероятностный характер и не гарантирует безопасность.",
}


@dataclass
class RiskResult:
    risk_score: int
    risk_level: str          # "LOW" | "MEDIUM" | "HIGH"
    risk_level_lower: str    # "low" | "medium" | "high" (для совместимости с БД/шаблонами)
    category: str            # "SAFE" | "SUSPICIOUS" | "FRAUDULENT" (или DANGEROUS для URL)
    legacy_category: str     # "likely_safe" | "suspicious" | "phishing"
    reasons: list[str]
    recommendation: str
    disclaimer: str
    components: dict

    def to_dict(self) -> dict:
        return asdict(self)


def load_risk_config(config_path: Path | str | None = None) -> dict:
    """Читает актуальный risk_config.yaml (без кэширования, чтобы правка файла сразу меняла расчёт)."""
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    for section, values in loaded.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def compute_risk(
    p_sms: float | None = None,
    p_url_max: float | None = None,
    se_groups: list[str] | set[str] | dict | int | None = None,
    url_flags: dict | list[dict] | None = None,
    config_path: Path | str | None = None,
    is_url_only: bool = False,
) -> RiskResult:
    """Рассчитывает Risk Score по формуле Задания 13:
    RiskScore = 100 * (w1 * P_sms + w2 * P_url_max + w3 * S_se) / (w1 + w2 * [URL есть] + w3),
    S_se = min(k / 3, 1), затем применяет правила усиления (override) из risk_config.yaml.
    """
    cfg = load_risk_config(config_path)
    w1 = float(cfg["weights"]["w1_sms"])
    w2 = float(cfg["weights"]["w2_url"])
    w3 = float(cfg["weights"]["w3_se"])
    k_max = max(1, int(cfg["se_normalization"].get("max_groups_for_full_score", 3)))

    if isinstance(se_groups, int):
        group_names = [f"GROUP_{i}" for i in range(max(0, se_groups))]
    elif isinstance(se_groups, dict):
        group_names = list(se_groups.keys())
    elif se_groups is None:
        group_names = []
    else:
        group_names = list(se_groups)

    k = len(group_names)
    s_se = min(k / float(k_max), 1.0)

    has_sms = p_sms is not None and not is_url_only
    has_url = p_url_max is not None

    p_sms_val = max(0.0, min(1.0, float(p_sms))) if p_sms is not None else 0.0
    p_url_val = max(0.0, min(1.0, float(p_url_max))) if p_url_max is not None else 0.0

    # Если анализируется только отдельный URL без текста SMS, оценка строится по URL-модели и флагам URL
    if is_url_only and has_url:
        base_score = 100.0 * p_url_val
        components = {"sms": 0.0, "url": round(base_score, 2), "rules": 0.0}
    else:
        eff_w1 = w1 if has_sms else 0.0
        eff_w2 = w2 if has_url else 0.0
        eff_w3 = w3
        denom = eff_w1 + eff_w2 + eff_w3
        if denom <= 0:
            base_score = 0.0
        else:
            numerator = eff_w1 * p_sms_val + eff_w2 * p_url_val + eff_w3 * s_se
            base_score = 100.0 * (numerator / denom)
        components = {
            "sms": round(100.0 * (eff_w1 * p_sms_val / denom), 2) if denom > 0 else 0.0,
            "url": round(100.0 * (eff_w2 * p_url_val / denom), 2) if denom > 0 and has_url else 0.0,
            "rules": round(100.0 * (eff_w3 * s_se / denom), 2) if denom > 0 else 0.0,
        }

    score = int(round(max(0.0, min(100.0, base_score))))

    # Сбор флагов URL (может быть передан один словарь признаков или список словарей)
    flag_dicts = []
    if isinstance(url_flags, dict):
        flag_dicts = [url_flags]
    elif isinstance(url_flags, list):
        flag_dicts = [d for d in url_flags if isinstance(d, dict)]

    has_ip = any(bool(d.get("has_ip")) for d in flag_dicts)
    has_punycode = any(bool(d.get("has_punycode")) for d in flag_dicts)
    has_brand_spoof = any(bool(d.get("brand_spoof") or d.get("has_brand_word")) for d in flag_dicts)
    has_shortener = any(bool(d.get("has_shortener")) for d in flag_dicts)

    ov = cfg["overrides"]
    t2 = float(cfg["thresholds"].get("url_dangerous_t2", 0.55))

    # Правила усиления (override) по Заданию 13 п.1:
    # 1. Группа CODE_REQUEST сработала -> Risk Score не ниже 70
    if "CODE_REQUEST" in group_names:
        score = max(score, int(ov.get("code_request_min_score", 70)))

    # 2. P_sms >= 0.8 и P_url_max >= t2 -> Risk Score не ниже 85
    if has_sms and has_url and p_sms_val >= float(ov.get("high_sms_threshold", 0.80)) and p_url_val >= t2:
        score = max(score, int(ov.get("high_sms_and_dangerous_url_min_score", 85)))

    # 3. URL с IP-адресом или Punycode/маскировкой под бренд -> Risk Score не ниже 60 (или 75 при наличии подозрительных слов)
    if has_ip or has_punycode or has_brand_spoof:
        score = max(score, int(ov.get("ip_or_punycode_or_brand_min_score", 60)))
        if any(d.get("suspicious_word_count", 0) > 0 for d in flag_dicts) or (has_sms and p_sms_val >= 0.6):
            score = max(score, int(cfg["thresholds"]["high_min"]))

    # 4. Совпадение нескольких групп социальной инженерии (>= 2 групп) или группа + высокий P_sms / сокращатель
    if k >= int(ov.get("multi_se_groups_count", 2)):
        score = max(score, int(ov.get("multi_se_min_score", 72)))
    elif k >= 1 and ((has_sms and p_sms_val >= 0.75) or has_shortener or (has_url and p_url_val >= t2)):
        score = max(score, int(cfg["thresholds"]["high_min"]))

    score = max(0, min(100, int(score)))

    med_min = int(cfg["thresholds"]["medium_min"])
    high_min = int(cfg["thresholds"]["high_min"])

    if score < med_min:
        risk_level = "LOW"
        category = "SAFE"
        legacy_cat = "likely_safe"
    elif score < high_min:
        risk_level = "MEDIUM"
        category = "SUSPICIOUS"
        legacy_cat = "suspicious"
    else:
        risk_level = "HIGH"
        category = "DANGEROUS" if is_url_only else "FRAUDULENT"
        legacy_cat = "phishing"

    # Генерация объяснений на русском языке, отсортированных по вкладу (не более 6 причин, Задание 13 п.5)
    weighted_reasons: list[tuple[float, str]] = []
    if has_sms and p_sms_val > 0.5:
        weighted_reasons.append((
            p_sms_val * 50.0,
            f"ML-модель текста оценила высокую вероятность мошенничества (P_sms = {p_sms_val:.2f})",
        ))
    if has_url and p_url_val > 0.5:
        weighted_reasons.append((
            p_url_val * 45.0,
            f"ML-модель ссылок выявила высокую вероятность фишинга (P_url = {p_url_val:.2f})",
        ))
    for g in group_names:
        title = GROUP_TITLES.get(g, g)
        w = 48.0 if g == "CODE_REQUEST" else 35.0
        weighted_reasons.append((w, f"Обнаружен признак социальной инженерии: {title.lower()} ({g})"))

    for d in flag_dicts:
        for r_text in explain_url(d):
            if "HTTP " in r_text and score < med_min:
                continue
            weighted_reasons.append((30.0, r_text))

    # Убираем дубликаты с сохранением порядка убывания вклада
    weighted_reasons.sort(key=lambda x: x[0], reverse=True)
    reasons: list[str] = []
    for _, text_r in weighted_reasons:
        if text_r not in reasons:
            reasons.append(text_r)
        if len(reasons) == 6:
            break

    if not reasons:
        if score < med_min:
            reasons.append("Явные признаки мошенничества, социальной инженерии и опасных ссылок не обнаружены")
        else:
            reasons.append(f"Совокупная вероятностная оценка риска составила {score}/100")

    rec = cfg["recommendations"].get(risk_level, DEFAULT_CONFIG["recommendations"][risk_level])
    disc = cfg.get("disclaimer", DEFAULT_CONFIG["disclaimer"])

    return RiskResult(
        risk_score=score,
        risk_level=risk_level,
        risk_level_lower=risk_level.lower(),
        category=category,
        legacy_category=legacy_cat,
        reasons=reasons,
        recommendation=rec,
        disclaimer=disc,
        components=components,
    )


def calculate_risk(p_sms=None, p_url=None, rule_groups=0, url_flags=None, config_path=None) -> dict:
    """Обёртка обратной совместимости, возвращающая словарь."""
    res = compute_risk(p_sms=p_sms, p_url_max=p_url, se_groups=rule_groups, url_flags=url_flags, config_path=config_path)
    return {
        "risk_score": res.risk_score,
        "risk_level": res.risk_level_lower,
        "risk_level_code": res.risk_level,
        "category": res.category,
        "legacy_category": res.legacy_category,
        "reasons": res.reasons,
        "recommendation": res.recommendation,
        "disclaimer": res.disclaimer,
        "components": res.components,
    }
