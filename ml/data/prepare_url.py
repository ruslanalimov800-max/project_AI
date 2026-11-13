"""Задание 10: Подготовка датасета URL, устранение артефактов разметки,
разбиение 70/15/15 по группам регистрируемого домена (GroupShuffleSplit),
построение матрицы признаков url_features.parquet и EDA.
"""
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from ml.url_features import _extract_tld_cached, _normalize_url, extract_url_features, validate_url

ROOT = Path(__file__).resolve().parents[2]
PHIUSIIL_CSV = ROOT / "data/raw/url_dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
KAGGLE_CSV = ROOT / "data/raw/url_dataset/malicious_phish.csv"
OUT_PARQUET = ROOT / "data/processed/url_features.parquet"
OUT_SPLIT = ROOT / "data/processed/url_split.json"
OUT_REPORT = ROOT / "data/processed/url_processing_report.json"

CORE_22_FEATURES = [
    "url_length", "host_length", "path_length", "dot_count", "subdomain_count",
    "domain_hyphen_count", "digit_count", "domain_digit_ratio", "special_char_count",
    "has_ip", "https", "non_standard_port", "has_at", "double_slash_in_path",
    "has_shortener", "suspicious_word_count", "has_punycode", "mixed_alphabet",
    "brand_spoof", "domain_entropy", "suspicious_tld", "query_param_count",
]


def canonicalize_url(raw_url: str) -> tuple[str, str]:
    """Нормализует регистр домена и возвращает (url, reg_domain)."""
    norm = _normalize_url(raw_url)
    try:
        parsed = urlsplit(norm)
    except ValueError:
        return "", ""
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return "", ""
    netloc = host
    try:
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
    except ValueError:
        pass
    if parsed.username:
        netloc = f"{parsed.username}@{netloc}"
    canon = urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "", parsed.query or "", ""))
    ext = _extract_tld_cached(host)
    reg_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host
    return canon, reg_domain


def prepare_url_dataset(sample_size: int = 60000, random_state: int = 42):
    """Загружает PhiUSIIL (основной) и дополняет путями из Malicious URLs (для устранения артефактов разметки),
    очищает, строит разбиение 70/15/15 по GroupShuffleSplit (по reg_domain) и сохраняет parquet.
    """
    phi_raw = pd.read_csv(PHIUSIIL_CSV, usecols=["URL", "label"])
    phi_total = len(phi_raw)
    # В PhiUSIIL: 1 — легитимный, 0 — фишинг. Приводим к стандарту ТЗ (Задание 10 п.2): 0 — безопасный, 1 — фишинг.
    phi_raw["label"] = (phi_raw["label"] == 0).astype(int)
    phi_raw = phi_raw.rename(columns={"URL": "url"})
    phi_raw["source"] = "PhiUSIIL"

    # В PhiUSIIL 100% легитимных URL имеют вид https://www.<domain> без пути (path_length == 0).
    # Чтобы устранить этот артефакт разметки (Задание 10 п.5) и научить модель корректно оценивать
    # безопасные ссылки с путями (например https://www.example.com/about) и без www.,
    # добавляем сбалансированную часть из второго открытого набора ТЗ (Malicious URLs Dataset, Kaggle),
    # приводя URL без схемы к https:// или http:// со стратификацией.
    frames = [phi_raw]
    kaggle_total = 0
    if KAGGLE_CSV.exists():
        kag = pd.read_csv(KAGGLE_CSV, usecols=["url", "type"]).dropna()
        kaggle_total = len(kag)
        # Объединение 4 классов в бинарную метку: benign -> 0, (phishing, malware, defacement) -> 1
        kag["label"] = (kag["type"] != "benign").astype(int)
        # Для benign без схемы в Kaggle добавляем https:// (75%) или http:// (25%), чтобы не создавать обратный артефакт схемы
        rng = np.random.default_rng(random_state)
        no_scheme = ~kag["url"].astype(str).str.match(r"(?i)^https?://")
        schemes = np.where(rng.random(len(kag)) < 0.75, "https://", "http://")
        kag.loc[no_scheme, "url"] = schemes[no_scheme] + kag.loc[no_scheme, "url"].astype(str)
        kag["source"] = "Kaggle_Malicious_URLs"
        frames.append(kag[["url", "label", "source"]])

    combined = pd.concat(frames, ignore_index=True)
    raw_combined_count = len(combined)

    # Удаление пустых и некорректных URL, нормализация регистра домена
    combined["url"] = combined["url"].astype(str).str.strip()
    combined = combined[combined["url"].ne("")].copy()

    # Стратифицированная подвыборка с запасом перед точной валидацией и дедупликацией
    if len(combined) > sample_size * 2:
        combined, _ = train_test_split(
            combined, train_size=sample_size * 2, stratify=combined["label"], random_state=random_state
        )

    valid_mask = combined["url"].map(validate_url)
    invalid_removed = int((~valid_mask).sum())
    combined = combined[valid_mask].copy()

    canon_pairs = combined["url"].map(canonicalize_url)
    combined["url_canon"] = [p[0] for p in canon_pairs]
    combined["reg_domain"] = [p[1] for p in canon_pairs]
    combined = combined[combined["url_canon"].ne("") & combined["reg_domain"].ne("")].copy()

    # Для нейтрализации артефакта "100% безопасных в PhiUSIIL начинаются с https://www."
    # у половины записей с префиксом "://www." убираем искусственный "www." в обоих классах одинаково
    rng = np.random.default_rng(random_state)
    has_www = combined["url_canon"].str.contains(r"^https?://www\.", regex=True)
    drop_www = has_www & (rng.random(len(combined)) < 0.5)
    combined.loc[drop_www, "url_canon"] = combined.loc[drop_www, "url_canon"].str.replace(
        r"^(https?://)www\.", r"\1", regex=True
    )

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["url_canon"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(combined)

    # Итоговая стратифицированная подвыборка 60 000 записей (Задание 10 п.4: допускается 50–100 тыс.)
    if len(combined) > sample_size:
        combined, _ = train_test_split(
            combined, train_size=sample_size, stratify=combined["label"], random_state=random_state
        )
        combined = combined.reset_index(drop=True)

    # Построение признаков единой функцией extract_url_features (Задание 9)
    feature_dicts = [extract_url_features(u) for u in combined["url_canon"]]
    feat_df = pd.DataFrame(feature_dicts)[CORE_22_FEATURES]
    full_df = pd.concat([combined[["url_canon", "reg_domain", "label", "source"]].rename(columns={"url_canon": "url"}), feat_df], axis=1)

    # Разбиение 70/15/15 по группам регистрируемого домена (GroupShuffleSplit, Задание 10 п.6)
    groups = full_df["reg_domain"].to_numpy()
    idx = np.arange(len(full_df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(idx, full_df["label"], groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_rel, test_rel = next(gss2.split(temp_idx, full_df["label"].iloc[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # Проверка отсутствия пересечения доменов
    train_domains = set(groups[train_idx])
    val_domains = set(groups[val_idx])
    test_domains = set(groups[test_idx])
    assert len(train_domains & test_domains) == 0, "Пересечение доменов между train и test!"
    assert len(train_domains & val_domains) == 0, "Пересечение доменов между train и val!"
    assert len(val_domains & test_domains) == 0, "Пересечение доменов между val и test!"

    full_df["split"] = "train"
    full_df.loc[val_idx, "split"] = "val"
    full_df.loc[test_idx, "split"] = "test"

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(OUT_PARQUET, index=False)

    split_data = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    OUT_SPLIT.write_text(json.dumps(split_data), encoding="utf-8")

    # EDA: сравнение 6 ключевых признаков по классам (Задание 10 п.8)
    key_6 = ["url_length", "host_length", "path_length", "subdomain_count", "suspicious_word_count", "domain_entropy"]
    eda_means = {}
    for col in key_6:
        eda_means[col] = {
            "safe_mean": round(float(full_df.loc[full_df["label"] == 0, col].mean()), 4),
            "phishing_mean": round(float(full_df.loc[full_df["label"] == 1, col].mean()), 4),
            "safe_median": round(float(full_df.loc[full_df["label"] == 0, col].median()), 4),
            "phishing_median": round(float(full_df.loc[full_df["label"] == 1, col].median()), 4),
        }

    report = {
        "phiusiil_raw_rows": phi_total,
        "kaggle_raw_rows": kaggle_total,
        "combined_raw_rows": raw_combined_count,
        "invalid_removed": invalid_removed,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(full_df),
        "class_balance": {
            "safe_0": int((full_df["label"] == 0).sum()),
            "phishing_1": int((full_df["label"] == 1).sum()),
            "phishing_share": round(float(full_df["label"].mean()), 4),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "unique_domains": {
            "train": len(train_domains),
            "val": len(val_domains),
            "test": len(test_domains),
            "train_test_overlap": len(train_domains & test_domains),
        },
        "eda_key_6_features": eda_means,
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_df, report


if __name__ == "__main__":
    _, rep = prepare_url_dataset()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
