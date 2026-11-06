from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response


def home(request):
    """Главная страница-заглушка (задание 2)."""
    return render(request, 'index.html')


@api_view(['GET'])
def health(request):
    """GET /api/v1/health/ — статус приложения."""
    return Response({'status': 'ok'})
