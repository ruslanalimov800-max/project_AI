from django.contrib.auth.decorators import login_required
from django.db.models import Count, Avg
from django.shortcuts import render
from apps.analysis.models import Check

@login_required
def dashboard(request):
    qs = Check.objects.filter(user=request.user)
    summary = qs.aggregate(total=Count("id"), average_risk=Avg("risk_score"))
    levels = list(qs.values("risk_level").annotate(count=Count("id")).order_by("risk_level"))
    recent = qs[:10]
    return render(request, "dashboard.html", {"summary": summary, "levels": levels, "recent": recent})
