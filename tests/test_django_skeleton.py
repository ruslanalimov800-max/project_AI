"""Задание 2: каркас Django запускается, главная и health отвечают, логирование настроено."""
import logging

import pytest
from django.conf import settings


def test_home_page_uses_bootstrap5_and_navigation(client):
    response = client.get('/')
    html = response.content.decode()
    assert response.status_code == 200
    assert 'bootstrap@5' in html and 'navbar' in html


def test_health_endpoint(client):
    response = client.get('/api/v1/health/')
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_installed_apps_follow_structure():
    for name in ('apps.accounts', 'apps.analysis', 'apps.dashboard'):
        assert name in settings.INSTALLED_APPS


def test_log_rotation_5_files_1mb():
    handler = logging.getLogger('sms_guard').handlers[0]
    assert handler.maxBytes == 1024 * 1024 and handler.backupCount == 5
    assert handler.baseFilename.replace('\\', '/').endswith('logs/app.log')


def test_missing_secret_key_without_debug_is_refused(monkeypatch):
    """При DEBUG=0 и без SECRET_KEY приложение не стартует с небезопасным ключом по умолчанию."""
    import importlib

    from django.core.exceptions import ImproperlyConfigured
    import config.settings as cfg

    monkeypatch.setenv('DEBUG', '0')
    monkeypatch.setenv('SECRET_KEY', '')
    with pytest.raises(ImproperlyConfigured):
        importlib.reload(cfg)
    monkeypatch.setenv('SECRET_KEY', 'restored-for-other-tests')
    importlib.reload(cfg)
