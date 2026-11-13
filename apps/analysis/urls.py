from django.urls import path
from . import views
urlpatterns = [
    path("", views.home, name="home"),
    path("api/v1/health/", views.health, name="health"),
    path("analyze/", views.analyze_page, name="analyze"),
    path("api/v1/analyze/", views.analyze_api, name="analyze-api"),
    path("history/", views.history, name="history"),
]
