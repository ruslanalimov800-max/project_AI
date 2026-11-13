import json
import re
from datetime import timedelta
from urllib.parse import urlsplit
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from ml.analyzer import analyze_message
from ml.url_features import extract_url_features
from .models import Check, URLResult, DetectedFeature
from .serializers import AnalyzeSerializer

CATEGORY_LABELS = [("likely_safe", "Безопасное"), ("suspicious", "Подозрительное"), ("phishing", "Мошенническое")]

def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")

@api_view(["GET"])
def health(request):
    return Response({"status": "ok"})

def _mask(text):
    text = re.sub(r'(?i)\b\d{13,19}\b', '[номер скрыт]', str(text))
    text = re.sub(r'(?i)\b\d{4,8}\b', '[код скрыт]', text)
    return text[:10000]

def _save_check(user, result):
    with transaction.atomic():
        check = Check.objects.create(
            user=user, check_type=result["check_type"], input_text=result["input_text"],
            input_masked=_mask(result["input_text"]), language=result["language"],
            category=result["category"], risk_level=result["risk_level"],
            risk_score=result["risk_score"], p_sms=result["p_sms"], p_url=result["p_url_max"],
            explanation=result["explanation"], recommendation=result["recommendation"],
            sms_model_version="sms-baseline-v1", url_model_version="url-model" if result.get("url_model_available") else "url-not-trained"
        )
        for item in result["url_predictions"]:
            URLResult.objects.create(analysis_check=check, url=item["url"], p_phishing=item["probability"],
                label=("phishing" if item["label"] == 1 else "legitimate") if item["label"] is not None else "unscored")
        for group in result["detected_features"]:
            for phrase in group["phrases"]:
                DetectedFeature.objects.create(analysis_check=check, group=group["group"], phrase=phrase)
    return check

def _run_analysis(request, expected_type):
    result = None
    submitted = ""
    if request.method == "POST":
        submitted = request.POST.get("text", "").strip()
        try:
            if not submitted:
                raise ValueError("Введите текст для проверки.")
            if expected_type == "sms" and len(submitted) > 1000:
                raise ValueError("Сообщение не должно превышать 1000 символов.")
            result = analyze_message(submitted)
            if expected_type == "url" and result["check_type"] != "url":
                raise ValueError("Введите одну корректную ссылку, например https://example.com.")
            if expected_type == "sms" and result["check_type"] == "url":
                # URL можно проверять на странице SMS, но тип результата остаётся URL.
                pass
            check = _save_check(request.user, result)
            result["check_id"] = check.pk
            result["risk_display"] = check.get_risk_level_display()
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception:
            messages.error(request, "Не удалось выполнить анализ. Проверьте доступность ML-моделей.")
    return result, submitted

@login_required
def analyze_page(request):
    result, _ = _run_analysis(request, "sms")
    return render(request, "analysis.html", {"result": result, "active": "sms"})

@login_required
def analyze_url_page(request):
    result, submitted = _run_analysis(request, "url")
    features = extract_url_features(submitted) if result and submitted else {}
    feature_labels = {
        "url_length":"Длина URL", "host_length":"Длина домена", "path_length":"Длина пути", "query_length":"Длина параметров",
        "dot_count":"Количество точек", "hyphen_count":"Количество дефисов", "at_count":"Символ @", "https":"HTTPS",
        "has_ip":"IP вместо домена", "has_punycode":"Punycode", "has_shortener":"Сервис сокращения", "suspicious_tld":"Подозрительная зона",
        "has_brand_word":"Упоминание бренда", "has_login_word":"Слова входа/проверки", "has_redirect":"Параметр перенаправления",
        "subdomain_count":"Количество поддоменов", "digit_ratio":"Доля цифр", "special_ratio":"Доля спецсимволов"
    }
    feature_rows = [{"label": label, "value": features[key], "active": bool(features[key]) if key in {"https","has_ip","has_punycode","has_shortener","suspicious_tld","has_brand_word","has_login_word","has_redirect"} else False} for key,label in feature_labels.items() if key in features]
    domain = urlsplit(submitted if "://" in submitted else "http://" + submitted).hostname if submitted else ""
    if result:
        result["domain"] = domain
        result["feature_rows"] = feature_rows
    return render(request, "url_check.html", {"result": result, "submitted": submitted, "active": "url"})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_api(request):
    serializer = AnalyzeSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        result = analyze_message(serializer.validated_data["text"])
        check = _save_check(request.user, result)
        result["check_id"] = check.pk
        return Response(result, status=status.HTTP_200_OK)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

@login_required
def history(request):
    qs = Check.objects.filter(user=request.user)
    filters = {key: request.GET.get(key, "") for key in ("date_from", "date_to", "type", "category", "risk")}
    if filters["date_from"]: qs = qs.filter(created_at__date__gte=filters["date_from"])
    if filters["date_to"]: qs = qs.filter(created_at__date__lte=filters["date_to"])
    if filters["type"] in ("sms", "url"): qs = qs.filter(check_type=filters["type"])
    if filters["category"] in dict(CATEGORY_LABELS): qs = qs.filter(category=filters["category"])
    if filters["risk"] in ("low", "medium", "high"): qs = qs.filter(risk_level=filters["risk"])
    paginator = Paginator(qs.order_by("-created_at"), 20)
    page_obj = paginator.get_page(request.GET.get("page"))
    query = request.GET.copy(); query.pop("page", None)
    return render(request, "history.html", {"page_obj": page_obj, "filters": filters, "categories": CATEGORY_LABELS, "query_string": query.urlencode(), "active": "history"})

@login_required
def check_detail(request, pk):
    check = get_object_or_404(Check, pk=pk, user=request.user)
    return render(request, "check_detail.html", {"check": check, "active": "history"})
