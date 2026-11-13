from django.urls import path
from apps.accounts.views import login_view, logout_view, profile_view, register
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    # Маршруты по разделу 8.1 ТЗ
    path("login/", login_view, name="login-root"),
    path("register/", register, name="register-root"),
    path("logout/", logout_view, name="logout-root"),
    path("profile/", profile_view, name="profile-root"),
    path("check/sms/", views.analyze_page, name="check-sms"),
    path("check/url/", views.analyze_url_page, name="check-url"),
    path("stats/", views.stats_page, name="stats"),
    # Совместимые маршруты
    path("analyze/", views.analyze_page, name="analyze"),
    path("analyze/url/", views.analyze_url_page, name="analyze-url"),
    path("checks/<int:pk>/", views.check_detail, name="check-detail"),
    path("history/", views.history, name="history"),
    path("history/export/", views.export_history_csv, name="history-export"),
    # REST API v1 (Раздел 4.4 ТЗ)
    path("api/v1/health/", views.health, name="health"),
    path("api/v1/auth/register/", views.api_auth_register, name="api-auth-register"),
    path("api/v1/auth/login/", views.api_auth_login, name="api-auth-login"),
    path("api/v1/auth/logout/", views.api_auth_logout, name="api-auth-logout"),
    path("api/v1/check/sms/", views.api_check_sms, name="api-check-sms"),
    path("api/v1/check/url/", views.api_check_url, name="api-check-url"),
    path("api/v1/analyze/", views.analyze_api, name="analyze-api"),
    path("api/v1/checks/", views.api_checks_list, name="api-checks-list"),
    path("api/v1/checks/<int:pk>/", views.api_check_detail_or_delete, name="api-check-detail"),
    path("api/v1/stats/summary/", views.api_stats_summary, name="api-stats-summary"),
]
