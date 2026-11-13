import json
from datetime import timedelta
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import render
from django.utils import timezone
from apps.analysis.models import Check

@login_required
def dashboard(request):
    qs = Check.objects.filter(user=request.user)
    total = qs.count()
    safe_count = qs.filter(risk_level="low").count()
    suspicious_count = qs.filter(risk_level="medium").count()
    fraudulent_count = qs.filter(risk_level="high").count()
    today = timezone.localdate()
    start = today - timedelta(days=29)
    daily = qs.filter(created_at__date__gte=start).values("created_at__date").annotate(count=Count("id"))
    daily_map = {str(row["created_at__date"]): row["count"] for row in daily}
    days = [start + timedelta(days=i) for i in range(30)]
    recent = qs.order_by("-created_at")[:10]
    return render(request, "dashboard.html", {
        "total": total, "safe_count": safe_count, "suspicious_count": suspicious_count,
        "fraudulent_count": fraudulent_count, "day_labels": json.dumps([d.strftime("%d.%m") for d in days]),
        "day_values": json.dumps([daily_map.get(str(d), 0) for d in days]), "recent": recent, "active": "dashboard"
    })
