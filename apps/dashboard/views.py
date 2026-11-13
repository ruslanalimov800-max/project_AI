import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from apps.analysis.views import compute_dashboard_summary


@login_required
def dashboard(request):
    try:
        days = int(request.GET.get("days", 30))
    except ValueError:
        days = 30
    summary = compute_dashboard_summary(request.user, days_period=days)
    recent = summary["recent_checks"]
    return render(request, "dashboard.html", {
        **summary,
        "recent": recent,
        "day_labels": json.dumps(summary["day_labels"], ensure_ascii=False),
        "day_values": json.dumps(summary["day_values"]),
        "day_safe_json": json.dumps(summary["day_safe"]),
        "day_susp_json": json.dumps(summary["day_suspicious"]),
        "day_fraud_json": json.dumps(summary["day_fraudulent"]),
        "top_features_labels": json.dumps([f["group"] for f in summary["top_features"]], ensure_ascii=False),
        "top_features_values": json.dumps([f["count"] for f in summary["top_features"]]),
        "active": "dashboard",
    })
