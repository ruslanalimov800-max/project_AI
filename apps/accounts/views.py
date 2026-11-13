"""Модуль регистрации, авторизации и профиля пользователя (Раздел 2.1, ФТ-1.1..ФТ-1.5, Задание 16 п. 7)."""
import re
from django import forms
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.cache import cache
from django.shortcuts import redirect, render

from apps.analysis.models import Check
from .models import UserProfile

User = get_user_model()
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")


def validate_registration_fields(username: str, email: str, password: str) -> tuple[bool, str]:
    """Валидация ФТ-1.1 и ФТ-1.2:
    - логин: 3–30 символов, латиница, цифры, _; уникален;
    - e-mail: корректный и уникален;
    - пароль: не менее 8 символов, содержит и буквы, и цифры.
    """
    username = (username or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not USERNAME_RE.match(username):
        return False, "Логин должен содержать от 3 до 30 символов (только латинские буквы, цифры и символ подчёркивания _)."
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return False, "Введите корректный адрес электронной почты (e-mail)."
    if len(password) < 8 or not re.search(r"[A-Za-zА-Яа-яЁё]", password) or not re.search(r"\d", password):
        return False, "Пароль должен содержать не менее 8 символов и включать как буквы, так и цифры."
    if User.objects.filter(username__iexact=username).exists():
        return False, "Пользователь с таким логином уже зарегистрирован."
    if User.objects.filter(email__iexact=email).exists():
        return False, "Пользователь с таким e-mail уже зарегистрирован."
    return True, ""


def _lockout_keys(username: str, ip: str) -> tuple[str, str]:
    uname = (username or "anon").strip().lower()
    return f"auth_fails:{uname}:{ip}", f"auth_lock:{uname}:{ip}"


def get_client_ip(request) -> str:
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "127.0.0.1")


def check_login_lockout(username: str, ip: str) -> bool:
    _, lock_key = _lockout_keys(username, ip)
    return bool(cache.get(lock_key))


def record_failed_login(username: str, ip: str) -> int:
    """После 5 неудачных попыток подряд устанавливает паузу 60 секунд (ФТ-1.3)."""
    fail_key, lock_key = _lockout_keys(username, ip)
    fails = int(cache.get(fail_key, 0)) + 1
    cache.set(fail_key, fails, timeout=300)
    if fails >= 5:
        cache.set(lock_key, True, timeout=60)
    return fails


def clear_failed_login(username: str, ip: str):
    fail_key, lock_key = _lockout_keys(username, ip)
    cache.delete_many([fail_key, lock_key])


class RegistrationForm(forms.Form):
    username = forms.CharField(
        label="Логин (3–30 символов: латиница, цифры, _)",
        max_length=30,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "например, ruslan_01"}),
    )
    email = forms.EmailField(
        label="E-mail",
        widget=forms.EmailInput(attrs={"class": "form-control", "placeholder": "user@example.com"}),
    )
    password1 = forms.CharField(
        label="Пароль (не менее 8 символов, буквы и цифры)",
        widget=forms.PasswordInput(attrs={"class": "form-control"}),
    )
    password2 = forms.CharField(
        label="Подтверждение пароля",
        widget=forms.PasswordInput(attrs={"class": "form-control"}),
    )

    def clean(self):
        cleaned = super().clean()
        u = cleaned.get("username", "")
        e = cleaned.get("email", "")
        p1 = cleaned.get("password1", "")
        p2 = cleaned.get("password2", "")
        if p1 != p2:
            self.add_error("password2", "Введённые пароли не совпадают.")
            return cleaned
        ok, err = validate_registration_fields(u, e, p1)
        if not ok:
            if "Логин" in err or "логином" in err:
                self.add_error("username", err)
            elif "e-mail" in err or "почты" in err:
                self.add_error("email", err)
            else:
                self.add_error("password1", err)
        return cleaned

    def save(self):
        u = self.cleaned_data["username"].strip()
        e = self.cleaned_data["email"].strip().lower()
        p = self.cleaned_data["password1"]
        return User.objects.create_user(username=u, email=e, password=p)


def register(request):
    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("analyze")
    return render(request, "register.html", {"form": form})


def login_view(request):
    error_message = None
    username_val = ""
    if request.method == "POST":
        username_val = (request.POST.get("username") or "").strip()
        password_val = request.POST.get("password") or ""
        ip = get_client_ip(request)
        if check_login_lockout(username_val, ip):
            error_message = "Превышено число попыток входа (5 неудачных попыток подряд). Подождите 60 секунд."
        else:
            user = authenticate(request, username=username_val, password=password_val)
            if user is not None:
                clear_failed_login(username_val, ip)
                login(request, user)
                next_url = request.GET.get("next") or request.POST.get("next") or "analyze"
                return redirect(next_url)
            fails = record_failed_login(username_val, ip)
            if fails >= 5:
                error_message = "Превышено число попыток входа (5 неудачных попыток подряд). Вход заблокирован на 60 секунд."
            else:
                error_message = f"Неверный логин или пароль (попытка {fails} из 5)."
    return render(request, "login.html", {"error_message": error_message, "username": username_val})


def logout_view(request):
    logout(request)
    request.session.flush()
    return redirect("login")


@login_required
def profile_view(request):
    """Страница профиля (Задание 16 п. 7): согласие на хранение текста, смена пароля, удаление всей истории."""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    pwd_form = PasswordChangeForm(request.user)

    if request.method == "POST":
        action = request.POST.get("action", "")
        if action == "update_privacy":
            profile.store_sms_text = bool(request.POST.get("store_sms_text"))
            lang = request.POST.get("preferred_language", "ru")
            if lang in ("ru", "kk", "en"):
                profile.preferred_language = lang
            profile.save()
            messages.success(request, "Настройки приватности и профиля сохранены.")
            return redirect("profile")
        elif action == "change_password":
            pwd_form = PasswordChangeForm(request.user, request.POST)
            if pwd_form.is_valid():
                user = pwd_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Пароль успешно изменён.")
                return redirect("profile")
            else:
                messages.error(request, "Исправьте ошибки в форме смены пароля.")
        elif action == "delete_all_history":
            deleted_count, _ = Check.objects.filter(user=request.user).delete()
            messages.success(request, f"Вся ваша история проверок удалена ({deleted_count} записей).")
            return redirect("profile")

    check_count = Check.objects.filter(user=request.user).count()
    return render(request, "profile.html", {
        "profile": profile,
        "pwd_form": pwd_form,
        "check_count": check_count,
        "active": "profile",
    })
