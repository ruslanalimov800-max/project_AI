"""Настройки для pytest: тесты не зависят от файла .env, DEBUG остаётся выключенным."""
import os

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key')

from config.settings import *  # noqa: E402,F401,F403
