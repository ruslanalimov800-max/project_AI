import re
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from ml.analyzer import analyze_message
from .models import Check, URLResult, DetectedFeature
from .serializers import AnalyzeSerializer

def home(request):
    return render(request, "index.html")

@api_view(["GET"])
def health(request):
    return Response({"status": "ok"})

def _mask(text):
    text = re.sub(r'(?i)\b\d{12,19}\b', '[номер скрыт]', str(text))
    text = re.sub(r'(?i)\b\d{4,8}\b', '[код скрыт]', text)
    return text[:10000]

def _save_check(user, result):
    check = Check.objects.create(
        user=user, check_type=result["check_type"], input_text=result["input_text"],
        input_masked=_mask(result["input_text"]), language=result["language"],
        category=result["category"], risk_level=result["risk_level"],
        risk_score=result["risk_score"], p_sms=result["p_sms"], p_url=result["p_url_max"],
        explanation=result["explanation"], recommendation=result["recommendation"],
        sms_model_version="sms-baseline-v1", url_model_version="url-not-trained"
    )
    for item in result["url_predictions"]:
        URLResult.objects.create(check=check, url=item["url"], p_phishing=item["probability"],
                                 label=("phishing" if item["label"] == 1 else "legitimate") if item["label"] is not None else "unscored")
    for group in result["detected_features"]:
        for phrase in group["phrases"]:
            DetectedFeature.objects.create(check=check, group=group["group"], phrase=phrase)
    return check

@login_required
def analyze_page(request):
    result = None
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        try:
            result = analyze_message(text)
            check = _save_check(request.user, result)
            result["check_id"] = check.pk
        except Exception as exc:
            messages.error(request, str(exc))
    return render(request, "analysis.html", {"result": result})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_api(request):
    serializer = AnalyzeSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        result = analyze_message(serializer.validated_data["text"])
        _save_check(request.user, result)
        return Response(result, status=status.HTTP_200_OK)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

@login_required
def history(request):
    checks = Check.objects.filter(user=request.user).order_by("-created_at")[:100]
    return render(request, "history.html", {"checks": checks})
