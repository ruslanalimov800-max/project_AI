"""Views, REST API (Раздел 4.4), маскирование ПДн (Задание 16 п. 4), история и экспорт CSV."""
import csv
from datetime import timedelta
import hashlib
import json
import logging
import re
import time
from urllib.parse import urlsplit

from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from apps.accounts.models import UserProfile
from apps.accounts.views import (
    check_login_lockout,
    clear_failed_login,
    get_client_ip,
    record_failed_login,
    validate_registration_fields,
)
from ml.analyzer import analyze_message, analyze_sms_struct, analyze_url_struct
from ml.predictor import get_predictor
from ml.url_features import extract_url_features
from .models import Check, DetectedFeature, URLResult
from .serializers import SMSCheckSerializer, URLCheckSerializer, strip_control_chars

logger = logging.getLogger("sms_guard")
User = get_user_model()

CATEGORY_LABELS = [
    ("SAFE", "Безопасное"),
    ("SUSPICIOUS", "Подозрительное"),
    ("FRAUDULENT", "Мошенническое"),
    ("DANGEROUS", "Опасная ссылка"),
]

# Регулярные выражения маскирования персональных данных (Задание 16 п. 4, Раздел 2.7):
# номера карт (13-19 цифр, включая пробелы/дефисы), телефоны и денежные суммы заменяются на ***
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
_MONEY_RE = re.compile(
    r"(?i)(?<!\w)(?:[$£€₸]\s*\d(?:[\d., ]*\d)?|\d(?:[\d., ]*\d)?\s*(?:руб(?:лей|ля|ль)?|тг|тенге|usd|eur|kzt|[$£€₸]))(?!\w)"
)


def mask_sensitive_text(text: str) -> str:
    """Заменяет номера карт (13-19 цифр), телефонов и суммы с валютой на *** (Задание 16 п. 4)."""
    val = str(text or "")
    val = _CARD_RE.sub("***", val)
    val = _PHONE_RE.sub("***", val)
    val = _MONEY_RE.sub("***", val)
    return val[:10000]


def _mask(text: str) -> str:
    return mask_sensitive_text(text)


def api_error(code: str, message: str, http_status: int) -> Response:
    """Единый формат ошибок REST API (Раздел 4.4 ТЗ): {"error": {"code": ..., "message": ...}}."""
    return Response({"error": {"code": code, "message": message}}, status=http_status)


def check_rate_limit(user_id: int) -> bool:
    """Не более 30 проверок в минуту на пользователя (НФТ-6, Задание 14 п. 5)."""
    bucket = int(time.time() // 60)
    key = f"throttle_check:{user_id}:{bucket}"
    count = int(cache.get(key, 0)) + 1
    cache.set(key, count, timeout=70)
    return count <= 30


def _deduplicate_check(user, check_type: str, raw_input: str):
    """Защита от двойного клика / повторной отправки той же формы в течение 3 секунд (НФТ-3)."""
    digest = hashlib.sha256(f"{user.pk}:{check_type}:{raw_input}".encode("utf-8")).hexdigest()[:24]
    dedup_key = f"dedup_submit:{digest}"
    existing_id = cache.get(dedup_key)
    if existing_id:
        existing = Check.objects.filter(pk=existing_id, user=user).first()
        if existing:
            return existing
    return None


def _remember_dedup(user, check_type: str, raw_input: str, check_id: int):
    digest = hashlib.sha256(f"{user.pk}:{check_type}:{raw_input}".encode("utf-8")).hexdigest()[:24]
    cache.set(f"dedup_submit:{digest}", check_id, timeout=3)


def _save_check(user, result_dict: dict) -> Check:
    """Сохраняет результат проверки в транзакции с учётом согласия на хранение текста (Задание 16 п. 3-4)."""
    raw_input = result_dict["input_text"]
    check_type = result_dict.get("type") or result_dict.get("check_type", "SMS").upper()

    dup = _deduplicate_check(user, check_type, raw_input)
    if dup is not None:
        return dup

    profile, _ = UserProfile.objects.get_or_create(user=user)
    # Текст SMS сохраняется ТОЛЬКО при согласии пользователя (store_sms_text=True), иначе NULL
    stored_text = raw_input if profile.store_sms_text else None
    masked_text = mask_sensitive_text(raw_input)
    versions = result_dict.get("model_versions") or {"sms": "1.0.0", "url": "1.0.0"}

    with transaction.atomic():
        check = Check.objects.create(
            user=user,
            check_type=result_dict.get("check_type", "sms"),
            input_text=stored_text,
            input_masked=masked_text,
            language=result_dict.get("language", "en"),
            category=result_dict["category"],
            risk_level=result_dict["risk_level"],
            risk_score=result_dict["risk_score"],
            p_sms=result_dict.get("p_sms"),
            p_url=result_dict.get("p_url_max"),
            explanation=result_dict.get("explanation", []),
            recommendation=result_dict.get("recommendation", ""),
            sms_model_version=versions.get("sms", "1.0.0"),
            url_model_version=versions.get("url", "1.0.0"),
        )
        url_items = result_dict.get("url_details") or result_dict.get("url_predictions") or []
        for item in url_items:
            URLResult.objects.create(
                analysis_check=check,
                url=item["url"],
                domain=item.get("domain", ""),
                p_url=item.get("p_url", item.get("probability")),
                p_phishing=item.get("probability"),
                category=item.get("category", "SAFE"),
                label=("phishing" if item.get("label") == 1 else "legitimate") if item.get("label") is not None else "unscored",
                features_json=item.get("features", {}),
            )
        for group in result_dict.get("detected_features", []):
            phrases = group.get("phrases") or [""]
            for phrase in phrases:
                DetectedFeature.objects.create(
                    analysis_check=check,
                    code=group.get("code") or group.get("group", ""),
                    group=group.get("group", ""),
                    title=group.get("title", group.get("group", "")),
                    phrase=phrase,
                    weight=float(group.get("weight", 15.0)),
                )

    _remember_dedup(user, check_type, raw_input, check.pk)
    return check


def serialize_check_record(check: Check) -> dict:
    """Форматирует объект Check в точном соответствии с разделом 4.4 ТЗ."""
    url_objs = [
        {
            "url": u.url,
            "domain": u.domain,
            "p_url": u.p_url if u.p_url is not None else u.p_phishing,
            "category": u.category or ("DANGEROUS" if u.label == "phishing" else "SAFE"),
            "features": u.features_json or {},
        }
        for u in check.url_results.all()
    ]
    feat_codes = []
    for df in check.detected_features.all():
        c = df.code or df.group
        if c and c not in feat_codes:
            feat_codes.append(c)

    return {
        "id": check.pk,
        "check_id": check.pk,
        "type": check.check_type.upper(),
        "check_type": check.check_type.lower(),
        "input_masked": check.input_masked,
        "input_text": check.input_text,
        "language": check.language,
        "category": check.category,
        "risk_level": check.risk_level.upper(),
        "risk_level_lower": check.risk_level.lower(),
        "risk_score": check.risk_score,
        "p_sms": check.p_sms,
        "p_url_max": check.p_url,
        "urls": url_objs,
        "features": feat_codes,
        "explanation": check.explanation,
        "recommendation": check.recommendation,
        "disclaimer": "Оценка носит вероятностный характер и не гарантирует безопасность.",
        "model_versions": {
            "sms": check.sms_model_version or "1.0.0",
            "url": check.url_model_version or "1.0.0",
        },
        "created_at": check.created_at.isoformat(),
    }


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "index.html")


@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
    predictor = get_predictor()
    return Response({
        "status": "ok",
        "models": {
            "sms_loaded": predictor.sms_available,
            "url_loaded": predictor.url_available,
            "sms_version": predictor.sms_version,
            "url_version": predictor.url_version,
        },
    })


# --- REST API Аутентификации (Раздел 4.4) ---

@api_view(["POST"])
@permission_classes([AllowAny])
def api_auth_register(request):
    username = (request.data.get("username") or "").strip()
    email = (request.data.get("email") or "").strip().lower()
    password = request.data.get("password") or ""
    ok, err = validate_registration_fields(username, email, password)
    if not ok:
        return api_error("VALIDATION_ERROR", err, status.HTTP_400_BAD_REQUEST)
    user = User.objects.create_user(username=username, email=email, password=password)
    login(request, user)
    return Response(
        {"id": user.pk, "username": user.username, "email": user.email},
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
@permission_classes([AllowAny])
def api_auth_login(request):
    username = (request.data.get("username") or "").strip()
    password = request.data.get("password") or ""
    ip = get_client_ip(request)
    if check_login_lockout(username, ip):
        return api_error(
            "ACCOUNT_LOCKED",
            "После 5 неудачных попыток вход временно приостановлен на 60 секунд.",
            status.HTTP_429_TOO_MANY_REQUESTS,
        )
    user = authenticate(request, username=username, password=password)
    if user is None:
        fails = record_failed_login(username, ip)
        if fails >= 5:
            return api_error(
                "ACCOUNT_LOCKED",
                "Превышено 5 неудачных попыток входа. Пауза 60 секунд.",
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
        return api_error("INVALID_CREDENTIALS", "Неверный логин или пароль.", status.HTTP_401_UNAUTHORIZED)
    clear_failed_login(username, ip)
    login(request, user)
    return Response({"id": user.pk, "username": user.username, "email": user.email}, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([AllowAny])
def api_auth_logout(request):
    logout(request)
    request.session.flush()
    return Response(status=status.HTTP_204_NO_CONTENT)


# --- REST API Анализа SMS и URL (Раздел 4.4, Задание 14) ---

@api_view(["POST"])
@permission_classes([AllowAny])
def api_check_sms(request):
    if not request.user or not request.user.is_authenticated:
        return api_error("UNAUTHORIZED", "Требуется авторизация для выполнения проверки.", status.HTTP_401_UNAUTHORIZED)
    if not check_rate_limit(request.user.pk):
        return api_error("RATE_LIMIT_EXCEEDED", "Превышен лимит: не более 30 проверок в минуту.", status.HTTP_429_TOO_MANY_REQUESTS)

    predictor = get_predictor()
    if not predictor.sms_available:
        return api_error("MODEL_UNAVAILABLE", "ML-модель классификации SMS временно недоступна.", status.HTTP_503_SERVICE_UNAVAILABLE)

    serializer = SMSCheckSerializer(data=request.data)
    if not serializer.is_valid():
        first_err = next(iter(serializer.errors.values()))[0]
        return api_error("VALIDATION_ERROR", str(first_err), status.HTTP_400_BAD_REQUEST)

    t0 = time.perf_counter()
    text = serializer.validated_data["text"]
    result_dict = analyze_message(text)
    check = _save_check(request.user, result_dict)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # НФТ-9 и Задание 14 п. 7: в лог пишутся только id проверки, категория и время обработки (текст SMS НЕ пишется)
    logger.info("check_id=%s type=SMS category=%s risk_score=%s duration_ms=%.2f", check.pk, check.category, check.risk_score, elapsed_ms)

    payload = serialize_check_record(check)
    payload["language_warning"] = result_dict.get("language_warning")
    return Response(payload, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([AllowAny])
def api_check_url(request):
    if not request.user or not request.user.is_authenticated:
        return api_error("UNAUTHORIZED", "Требуется авторизация для выполнения проверки.", status.HTTP_401_UNAUTHORIZED)
    if not check_rate_limit(request.user.pk):
        return api_error("RATE_LIMIT_EXCEEDED", "Превышен лимит: не более 30 проверок в минуту.", status.HTTP_429_TOO_MANY_REQUESTS)

    predictor = get_predictor()
    if not predictor.url_available:
        return api_error("MODEL_UNAVAILABLE", "ML-модель классификации URL временно недоступна.", status.HTTP_503_SERVICE_UNAVAILABLE)

    serializer = URLCheckSerializer(data=request.data)
    if not serializer.is_valid():
        first_err = next(iter(serializer.errors.values()))[0]
        return api_error("VALIDATION_ERROR", str(first_err), status.HTTP_400_BAD_REQUEST)

    t0 = time.perf_counter()
    url_val = serializer.validated_data["url"]
    res_struct = analyze_url_struct(url_val)
    res_dict = res_struct.to_dict()
    res_dict["url_details"] = res_struct.urls
    check = _save_check(request.user, res_dict)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info("check_id=%s type=URL category=%s risk_score=%s duration_ms=%.2f", check.pk, check.category, check.risk_score, elapsed_ms)

    payload = serialize_check_record(check)
    return Response(payload, status=status.HTTP_200_OK)


# Совместимый эндпоинт /api/v1/analyze/
@api_view(["POST"])
@permission_classes([AllowAny])
def analyze_api(request):
    return api_check_sms(request._request)


# --- REST API Истории и Статистики (Раздел 4.4) ---

def _filter_checks_queryset(qs, params):
    date_from = params.get("date_from", "")
    date_to = params.get("date_to", "")
    c_type = (params.get("type") or "").lower()
    category = params.get("category") or ""
    risk = (params.get("risk_level") or params.get("risk") or "").lower()

    if date_from:
        qs = qs.filter(created_at__date__gte=date_from)
    if date_to:
        qs = qs.filter(created_at__date__lte=date_to)
    if c_type in ("sms", "url"):
        qs = qs.filter(check_type__iexact=c_type)
    if category:
        if category in ("SAFE", "likely_safe"):
            qs = qs.filter(category__in=["SAFE", "likely_safe"])
        elif category in ("SUSPICIOUS", "suspicious"):
            qs = qs.filter(category__in=["SUSPICIOUS", "suspicious"])
        elif category in ("FRAUDULENT", "DANGEROUS", "phishing"):
            qs = qs.filter(category__in=["FRAUDULENT", "DANGEROUS", "phishing"])
    if risk in ("low", "medium", "high"):
        qs = qs.filter(risk_level__iexact=risk)
    return qs


@api_view(["GET"])
@permission_classes([AllowAny])
def api_checks_list(request):
    if not request.user or not request.user.is_authenticated:
        return api_error("UNAUTHORIZED", "Требуется авторизация.", status.HTTP_401_UNAUTHORIZED)
    qs = Check.objects.filter(user=request.user).prefetch_related("url_results", "detected_features")
    qs = _filter_checks_queryset(qs, request.GET)
    paginator = Paginator(qs.order_by("-created_at"), 20)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    return Response({
        "count": paginator.count,
        "num_pages": paginator.num_pages,
        "page": page_obj.number,
        "results": [serialize_check_record(c) for c in page_obj],
    })


@api_view(["GET", "DELETE"])
@permission_classes([AllowAny])
def api_check_detail_or_delete(request, pk: int):
    if not request.user or not request.user.is_authenticated:
        return api_error("UNAUTHORIZED", "Требуется авторизация.", status.HTTP_401_UNAUTHORIZED)
    # ФТ-1.7: пользователь видит и удаляет только свои записи; чужая запись -> 404
    check = Check.objects.filter(pk=pk, user=request.user).first()
    if check is None:
        return api_error("NOT_FOUND", "Запись проверки не найдена или принадлежит другому пользователю.", status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        check.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    return Response(serialize_check_record(check), status=status.HTTP_200_OK)


def compute_dashboard_summary(user, days_period: int = 30) -> dict:
    """Агрегация данных для Dashboard в БД через annotate, Count, TruncDate (Задание 17 п. 7, ФТ-1.9)."""
    days_period = days_period if days_period in (7, 30, 90) else 30
    is_admin = bool(user.is_staff or user.is_superuser)
    base_qs = Check.objects.all() if is_admin else Check.objects.filter(user=user)

    today = timezone.localdate()
    start_date = today - timedelta(days=days_period - 1)
    qs = base_qs.filter(created_at__date__gte=start_date)

    agg = base_qs.aggregate(
        total=Count("id"),
        sms_count=Count("id", filter=Q(check_type__iexact="sms")),
        url_count=Count("id", filter=Q(check_type__iexact="url")),
        safe_count=Count("id", filter=Q(risk_level="low")),
        suspicious_count=Count("id", filter=Q(risk_level="medium")),
        fraudulent_count=Count("id", filter=Q(risk_level="high")),
        avg_risk_score=Avg("risk_score"),
    )

    daily_rows = (
        qs.annotate(day=TruncDate("created_at"))
        .values("day", "risk_level")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    day_list = [start_date + timedelta(days=i) for i in range(days_period)]
    day_map_total = {str(d): 0 for d in day_list}
    day_map_safe = {str(d): 0 for d in day_list}
    day_map_susp = {str(d): 0 for d in day_list}
    day_map_fraud = {str(d): 0 for d in day_list}
    for row in daily_rows:
        d_str = str(row["day"])
        if d_str in day_map_total:
            c = row["count"]
            day_map_total[d_str] += c
            rl = (row["risk_level"] or "").lower()
            if rl == "low":
                day_map_safe[d_str] += c
            elif rl == "medium":
                day_map_susp[d_str] += c
            else:
                day_map_fraud[d_str] += c

    # Топ-5 сработавших признаков социальной инженерии (Задание 17 п. 4)
    feat_qs = DetectedFeature.objects.all() if is_admin else DetectedFeature.objects.filter(analysis_check__user=user)
    top_features_qs = (
        feat_qs.values("group")
        .annotate(count=Count("id"))
        .order_by("-count")[:5]
    )
    top_features = [{"group": r["group"], "count": r["count"]} for r in top_features_qs]

    # Для администратора не выводим чужие тексты SMS (ФТ-1.9)
    recent_qs = (Check.objects.filter(user=user) if not is_admin else Check.objects.all()).order_by("-created_at")[:10]
    predictor = get_predictor()

    return {
        "is_admin": is_admin,
        "period_days": days_period,
        "total": agg["total"] or 0,
        "sms_count": agg["sms_count"] or 0,
        "url_count": agg["url_count"] or 0,
        "safe_count": agg["safe_count"] or 0,
        "suspicious_count": agg["suspicious_count"] or 0,
        "fraudulent_count": agg["fraudulent_count"] or 0,
        "avg_risk_score": round(float(agg["avg_risk_score"] or 0.0), 1),
        "day_labels": [d.strftime("%d.%m") for d in day_list],
        "day_values": [day_map_total[str(d)] for d in day_list],
        "day_safe": [day_map_safe[str(d)] for d in day_list],
        "day_suspicious": [day_map_susp[str(d)] for d in day_list],
        "day_fraudulent": [day_map_fraud[str(d)] for d in day_list],
        "top_features": top_features,
        "recent_checks": recent_qs,
        "active_models": {
            "sms": predictor.sms_meta or {"version": predictor.sms_version, "algorithm": "LinearSVM"},
            "url": predictor.url_meta or {"version": predictor.url_version, "algorithm": "RandomForest"},
        },
    }


@api_view(["GET"])
@permission_classes([AllowAny])
def api_stats_summary(request):
    if not request.user or not request.user.is_authenticated:
        return api_error("UNAUTHORIZED", "Требуется авторизация.", status.HTTP_401_UNAUTHORIZED)
    try:
        days = int(request.GET.get("days", 30))
    except ValueError:
        days = 30
    data = compute_dashboard_summary(request.user, days_period=days)
    recent_serialized = []
    for c in data.pop("recent_checks"):
        recent_serialized.append({
            "id": c.pk,
            "type": c.check_type.upper(),
            "fragment": "[Скрыто для администратора]" if (data["is_admin"] and c.user_id != request.user.pk) else c.input_masked[:60],
            "category": c.category,
            "risk_level": c.risk_level.upper(),
            "risk_score": c.risk_score,
            "created_at": c.created_at.strftime("%d.%m.%Y %H:%M"),
        })
    data["recent"] = recent_serialized
    return Response(data, status=status.HTTP_200_OK)


# --- HTML-страницы приложения (Раздел 8.1, Задания 15–17) ---

def _run_analysis(request, expected_type: str):
    result = None
    submitted = ""
    error_text = None
    if request.method == "POST":
        raw_val = request.POST.get("text") or request.POST.get("url") or ""
        submitted = strip_control_chars(raw_val).strip()
        predictor = get_predictor()
        try:
            if not check_rate_limit(request.user.pk):
                raise ValueError("Превышен лимит запросов (не более 30 проверок в минуту). Подождите немного.")
            if expected_type == "sms" and not predictor.sms_available:
                raise RuntimeError("Анализ временно недоступен: файл ML-модели SMS не загружен.")
            if expected_type == "url" and not predictor.url_available:
                raise RuntimeError("Анализ временно недоступен: файл ML-модели URL не загружен.")
            if not submitted:
                raise ValueError("Поле ввода не может быть пустым. Введите текст или ссылку.")
            if expected_type == "sms" and len(submitted) > 1000:
                raise ValueError("Текст SMS не должен превышать 1000 символов.")
            if expected_type == "url" and len(submitted) > 2048:
                raise ValueError("Длина URL не должна превышать 2048 символов.")

            if expected_type == "url":
                res_struct = analyze_url_struct(submitted)
                result = res_struct.to_dict()
                result["url_details"] = res_struct.urls
            else:
                result = analyze_message(submitted)

            check = _save_check(request.user, result)
            result["check_id"] = check.pk
            result["risk_display"] = check.get_risk_level_display()
        except ValueError as exc:
            error_text = str(exc)
            messages.error(request, error_text)
        except RuntimeError as exc:
            error_text = str(exc)
            messages.error(request, error_text)
        except Exception as exc:
            logger.exception("Unexpected analysis error: %s", exc)
            error_text = "Анализ временно недоступен. Проверьте состояние ML-моделей."
            messages.error(request, error_text)
    return result, submitted, error_text


@login_required
def analyze_page(request):
    predictor = get_predictor()
    result, submitted, error_text = _run_analysis(request, "sms")
    return render(request, "analysis.html", {
        "result": result,
        "submitted": submitted,
        "error_text": error_text,
        "model_available": predictor.sms_available,
        "active": "sms",
    })


@login_required
def analyze_url_page(request):
    predictor = get_predictor()
    result, submitted, error_text = _run_analysis(request, "url")
    features = extract_url_features(submitted) if result and submitted else {}
    feature_labels = {
        "url_length": "Длина URL",
        "host_length": "Длина домена (host)",
        "path_length": "Длина пути",
        "dot_count": "Число точек в URL",
        "subdomain_count": "Число поддоменов (tldextract)",
        "domain_hyphen_count": "Число дефисов в домене",
        "digit_count": "Число цифр в URL",
        "domain_digit_ratio": "Доля цифр в домене",
        "special_char_count": "Число спецсимволов (@~%=&_?)",
        "has_ip": "IP-адрес вместо домена",
        "https": "Схема HTTPS",
        "non_standard_port": "Нестандартный порт",
        "has_at": "Символ @ в URL",
        "double_slash_in_path": "Двойной // в пути",
        "has_shortener": "Сервис сокращения URL",
        "suspicious_word_count": "Число подозрительных слов",
        "has_punycode": "Punycode (xn--) в домене",
        "mixed_alphabet": "Смешение алфавитов",
        "brand_spoof": "Маскировка под бренд (Левенштейн ≤ 2)",
        "domain_entropy": "Энтропия Шеннона домена",
        "suspicious_tld": "Подозрительная доменная зона",
        "query_param_count": "Число параметров запроса",
    }
    flag_keys = {
        "has_ip", "non_standard_port", "has_at", "double_slash_in_path",
        "has_shortener", "suspicious_word_count", "has_punycode",
        "mixed_alphabet", "brand_spoof", "suspicious_tld",
    }
    feature_rows = [
        {
            "key": key,
            "label": label,
            "value": features[key],
            "active": bool(features[key]) if key in flag_keys else False,
        }
        for key, label in feature_labels.items()
        if key in features
    ]
    domain = urlsplit(submitted if "://" in submitted else "http://" + submitted).hostname if submitted else ""
    if result:
        result["domain"] = domain
        result["feature_rows"] = feature_rows
    return render(request, "url_check.html", {
        "result": result,
        "submitted": submitted,
        "error_text": error_text,
        "model_available": predictor.url_available,
        "active": "url",
    })


@login_required
def history(request):
    qs = Check.objects.filter(user=request.user)
    filters = {
        key: request.GET.get(key, "")
        for key in ("date_from", "date_to", "type", "category", "risk", "risk_level")
    }
    if not filters["risk"] and filters["risk_level"]:
        filters["risk"] = filters["risk_level"]
    qs = _filter_checks_queryset(qs, filters)
    paginator = Paginator(qs.order_by("-created_at"), 20)
    page_obj = paginator.get_page(request.GET.get("page"))
    query = request.GET.copy()
    query.pop("page", None)
    return render(request, "history.html", {
        "page_obj": page_obj,
        "filters": filters,
        "categories": CATEGORY_LABELS,
        "query_string": query.urlencode(),
        "active": "history",
    })


@login_required
def export_history_csv(request):
    """Экспорт собственной истории проверок в CSV (Раздел 2.7 ТЗ)."""
    qs = _filter_checks_queryset(Check.objects.filter(user=request.user), request.GET).order_by("-created_at")
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="sms_guard_history.csv"'
    response.write("\ufeff")
    writer = csv.writer(response)
    writer.writerow(["ID", "Дата", "Тип", "Язык", "Категория", "Уровень риска", "Risk Score", "P_sms", "P_url", "Маскированный текст"])
    for c in qs:
        writer.writerow([
            c.pk,
            c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            c.check_type.upper(),
            c.language,
            c.category,
            c.risk_level.upper(),
            c.risk_score,
            c.p_sms if c.p_sms is not None else "",
            c.p_url if c.p_url is not None else "",
            c.input_masked,
        ])
    return response


@login_required
def check_detail(request, pk: int):
    check = get_object_or_404(Check, pk=pk, user=request.user)
    if request.method == "POST" and request.POST.get("action") == "delete":
        check.delete()
        messages.success(request, f"Запись проверки №{pk} удалена.")
        return redirect("history")
    return render(request, "check_detail.html", {"check": check, "active": "history"})


@login_required
def stats_page(request):
    """Страница расширенной статистики /stats/ (Раздел 8.1 п. 7)."""
    try:
        days = int(request.GET.get("days", 30))
    except ValueError:
        days = 30
    summary = compute_dashboard_summary(request.user, days_period=days)
    return render(request, "stats.html", {
        **summary,
        "day_labels_json": json.dumps(summary["day_labels"], ensure_ascii=False),
        "day_safe_json": json.dumps(summary["day_safe"]),
        "day_susp_json": json.dumps(summary["day_suspicious"]),
        "day_fraud_json": json.dumps(summary["day_fraudulent"]),
        "top_features_labels": json.dumps([f["group"] for f in summary["top_features"]], ensure_ascii=False),
        "top_features_values": json.dumps([f["count"] for f in summary["top_features"]]),
        "active": "stats",
    })
