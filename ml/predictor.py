"""Задание 12: Загрузка обеих моделей один раз при старте приложения (класс-синглтон ModelPredictor)."""
import json
import logging
from pathlib import Path
import joblib
import numpy as np

from ml.data.prepare_url import CORE_22_FEATURES
from ml.url_features import extract_url_features
from ml.url_model import categorize_url_probability

logger = logging.getLogger("sms_guard")
ROOT = Path(__file__).resolve().parents[1]

SMS_MODEL_PATH = ROOT / "ml/artifacts/sms_model.joblib"
SMS_FALLBACK_PATH = ROOT / "ml/artifacts/sms_best_model.joblib"
SMS_BASELINE_PATH = ROOT / "ml/artifacts/sms_baseline.joblib"
SMS_META_PATH = ROOT / "ml/artifacts/sms_model_meta.json"

URL_MODEL_PATH = ROOT / "ml/artifacts/url_model.joblib"
URL_META_PATH = ROOT / "ml/artifacts/url_model_meta.json"


class ModelPredictor:
    """Синглтон для однократной загрузки SMS- и URL-моделей за время жизни процесса (Задание 12, НФТ-14)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.sms_bundle = None
        self.url_bundle = None
        self.sms_meta = {}
        self.url_meta = {}
        self.load_models()
        self._initialized = True

    def load_models(self):
        """Загружает модели в память один раз и пишет событие в лог."""
        # 1. Загрузка SMS-модели
        sms_path = next((p for p in (SMS_MODEL_PATH, SMS_FALLBACK_PATH, SMS_BASELINE_PATH) if p.exists()), None)
        if sms_path is not None:
            try:
                loaded = joblib.load(sms_path)
                if isinstance(loaded, dict):
                    self.sms_bundle = loaded
                else:
                    self.sms_bundle = {
                        "pipeline": loaded,
                        "threshold": 0.5,
                        "model_name": "LogisticRegression",
                        "version": "1.0.0",
                    }
                if SMS_META_PATH.exists():
                    self.sms_meta = json.loads(SMS_META_PATH.read_text(encoding="utf-8"))
                logger.info("SMS model loaded once from %s (version=%s)", sms_path.name, self.sms_version)
            except Exception as exc:
                logger.error("Failed to load SMS model from %s: %s", sms_path, exc)
                self.sms_bundle = None
        else:
            logger.warning("SMS model artifact is missing! Analysis will report model unavailability (НФТ-14).")
            self.sms_bundle = None

        # 2. Загрузка URL-модели
        if URL_MODEL_PATH.exists():
            try:
                self.url_bundle = joblib.load(URL_MODEL_PATH)
                if URL_META_PATH.exists():
                    self.url_meta = json.loads(URL_META_PATH.read_text(encoding="utf-8"))
                logger.info("URL model loaded once from %s (version=%s)", URL_MODEL_PATH.name, self.url_version)
            except Exception as exc:
                logger.error("Failed to load URL model from %s: %s", URL_MODEL_PATH, exc)
                self.url_bundle = None
        else:
            logger.warning("URL model artifact is missing! URL ML scoring is unavailable (НФТ-14).")
            self.url_bundle = None

    @property
    def sms_available(self) -> bool:
        return self.sms_bundle is not None

    @property
    def url_available(self) -> bool:
        return self.url_bundle is not None

    @property
    def sms_version(self) -> str:
        if not self.sms_bundle:
            return "unavailable"
        return str(self.sms_meta.get("version") or self.sms_bundle.get("version") or "1.0.0")

    @property
    def url_version(self) -> str:
        if not self.url_bundle:
            return "unavailable"
        return str(self.url_meta.get("version") or self.url_bundle.get("version") or "1.0.0")

    def predict_sms(self, text: str) -> dict | None:
        if not self.sms_available:
            return None
        b = self.sms_bundle
        thr = float(b.get("threshold", 0.5))
        if "pipeline" in b:
            prob = float(b["pipeline"].predict_proba([str(text)])[0, 1])
        else:
            vec = b["features"].transform([str(text)])
            prob = float(b["classifier"].predict_proba(vec)[0, 1])
        return {
            "label": int(prob >= thr),
            "probability": round(prob, 4),
            "threshold": thr,
            "version": self.sms_version,
        }

    def predict_url(self, url: str) -> dict | None:
        if not self.url_available:
            return None
        b = self.url_bundle
        feat_names = b.get("feature_names", CORE_22_FEATURES)
        feats = extract_url_features(str(url))
        vec = np.asarray([[float(feats[k]) for k in feat_names]], dtype=float)
        raw_prob = float(b["classifier"].predict_proba(vec)[0, 1])

        structural_boost = 0.0
        if feats["has_ip"] or feats["has_punycode"] or feats["brand_spoof"]:
            structural_boost += 0.48
        if feats["has_shortener"] or feats["non_standard_port"] or feats["has_at"] or feats["double_slash_in_path"]:
            structural_boost += 0.35
        if feats["suspicious_word_count"] >= 1 and (not feats["https"] or feats["suspicious_word_count"] >= 2 or structural_boost > 0):
            structural_boost += min(0.40, 0.18 * feats["suspicious_word_count"])
        if feats["suspicious_tld"]:
            structural_boost += 0.25

        if structural_boost > 0:
            prob = round(min(0.99, max(raw_prob, min(0.96, raw_prob * 0.45 + structural_boost))), 4)
        else:
            prob = round(min(0.32, raw_prob * 0.75) if feats["https"] else raw_prob, 4)

        thresholds = b.get("thresholds", {"t1": 0.35, "t2": 0.60})
        t1, t2 = float(thresholds.get("t1", 0.35)), float(thresholds.get("t2", 0.60))
        cat = categorize_url_probability(prob, t1=t1, t2=t2)
        return {
            "url": str(url),
            "p_url": prob,
            "probability": prob,
            "label": int(prob >= t2),
            "category": cat,
            "version": self.url_version,
            "features": feats,
        }


def get_predictor() -> ModelPredictor:
    return ModelPredictor()
