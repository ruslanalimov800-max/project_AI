from django.urls import path
from . import views
urlpatterns = [
    path("", views.home, name="home"),
    path("api/v1/health/", views.health, name="health"),
    path("analyze/", views.analyze_page, name="analyze"),
    path("analyze/url/", views.analyze_url_page, name="analyze-url"),
    path("checks/<int:pk>/", views.check_detail, name="check-detail"),
    path("api/v1/analyze/", views.analyze_api, name="analyze-api"),
    path("history/", views.history, name="history"),
]
