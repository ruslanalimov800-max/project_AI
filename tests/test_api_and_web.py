import json
import pytest
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client
from unittest.mock import patch

from apps.accounts.models import UserProfile
from apps.analysis.models import Check


@pytest.fixture(autouse=True)
def clear_cache_before_each():
    cache.clear()
    yield
    cache.clear()


@pytest.mark.django_db
def test_register_validation_and_success():
    c = Client()
    bad_pw = c.post(
        "/api/v1/auth/register/",
        data=json.dumps({"username": "u1", "email": "u1@example.com", "password": "short"}),
        content_type="application/json",
    )
    assert bad_pw.status_code == 400
    assert bad_pw.json()["error"]["code"] == "VALIDATION_ERROR"

    bad_email = c.post(
        "/api/v1/auth/register/",
        data=json.dumps({"username": "u1", "email": "not-an-email", "password": "Password123"}),
        content_type="application/json",
    )
    assert bad_email.status_code == 400

    ok = c.post(
        "/api/v1/auth/register/",
        data=json.dumps({"username": "newuser1", "email": "newuser1@example.com", "password": "Password123"}),
        content_type="application/json",
    )
    assert ok.status_code == 201
    assert ok.json()["username"] == "newuser1"
    assert User.objects.filter(username="newuser1").exists()


@pytest.mark.django_db
def test_login_lockout_after_5_failed_attempts():
    User.objects.create_user(username="lockuser", email="lock@example.com", password="Password123")
    c = Client()
    for i in range(4):
        r = c.post(
            "/api/v1/auth/login/",
            data=json.dumps({"username": "lockuser", "password": "WrongPassword1"}),
            content_type="application/json",
        )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "INVALID_CREDENTIALS"

    r5 = c.post(
        "/api/v1/auth/login/",
        data=json.dumps({"username": "lockuser", "password": "WrongPassword1"}),
        content_type="application/json",
    )
    assert r5.status_code == 429
    assert r5.json()["error"]["code"] == "ACCOUNT_LOCKED"

    r6_valid = c.post(
        "/api/v1/auth/login/",
        data=json.dumps({"username": "lockuser", "password": "Password123"}),
        content_type="application/json",
    )
    assert r6_valid.status_code == 429
    assert r6_valid.json()["error"]["code"] == "ACCOUNT_LOCKED"


@pytest.mark.django_db
def test_logout_destroys_session():
    User.objects.create_user(username="u_out", email="out@example.com", password="Password123")
    c = Client()
    c.login(username="u_out", password="Password123")
    assert c.get("/api/v1/checks/").status_code == 200
    r = c.post("/api/v1/auth/logout/")
    assert r.status_code == 204
    assert c.get("/api/v1/checks/").status_code == 401


@pytest.mark.django_db
def test_unauthorized_web_and_api_access():
    c = Client()
    assert c.get("/history/").status_code == 302
    assert c.get("/dashboard/").status_code == 302
    assert c.get("/profile/").status_code == 302
    assert c.post("/api/v1/check/sms/", data=json.dumps({"text": "Привет"}), content_type="application/json").status_code == 401
    assert c.post("/api/v1/check/url/", data=json.dumps({"url": "https://example.com"}), content_type="application/json").status_code == 401
    assert c.get("/api/v1/stats/summary/").status_code == 401


@pytest.mark.django_db
def test_api_check_sms_and_url_success():
    User.objects.create_user(username="tester", email="t@example.com", password="Password123")
    c = Client()
    c.login(username="tester", password="Password123")

    r_sms = c.post(
        "/api/v1/check/sms/",
        data=json.dumps({"text": "Срочно переведите деньги и сообщите код из SMS! https://bit.ly/demo-test"}),
        content_type="application/json",
    )
    assert r_sms.status_code == 200
    d_sms = r_sms.json()
    assert d_sms["category"] in ("SAFE", "SUSPICIOUS", "FRAUDULENT")
    assert 0 <= d_sms["risk_score"] <= 100
    assert len(d_sms["explanation"]) >= 1

    r_url = c.post(
        "/api/v1/check/url/",
        data=json.dumps({"url": "http://192.0.2.1/login/verify"}),
        content_type="application/json",
    )
    assert r_url.status_code == 200
    d_url = r_url.json()
    assert d_url["type"] == "URL"
    assert d_url["category"] in ("SAFE", "SUSPICIOUS", "DANGEROUS")


@pytest.mark.django_db
def test_api_validation_errors_no_500():
    User.objects.create_user(username="valuser", email="val@example.com", password="Password123")
    c = Client()
    c.login(username="valuser", password="Password123")

    for bad_text in ["", "   ", "a" * 1001, "ok\x00bad"]:
        r = c.post(
            "/api/v1/check/sms/",
            data=json.dumps({"text": bad_text}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    for bad_url in ["", "not a url", "javascript:alert(1)", "ftp://example.com/file"]:
        r = c.post(
            "/api/v1/check/url/",
            data=json.dumps({"url": bad_url}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.django_db
def test_foreign_check_isolation_404():
    u1 = User.objects.create_user(username="owner", email="o@example.com", password="Password123")
    User.objects.create_user(username="intruder", email="i@example.com", password="Password123")
    chk = Check.objects.create(
        user=u1,
        check_type="sms",
        input_masked="ok",
        risk_score=10,
        risk_level="low",
        category="SAFE",
        explanation=["ok"],
        recommendation="ok",
    )
    c = Client()
    c.login(username="intruder", password="Password123")
    assert c.get(f"/api/v1/checks/{chk.pk}/").status_code == 404
    assert c.delete(f"/api/v1/checks/{chk.pk}/").status_code == 404
    assert c.get(f"/checks/{chk.pk}/").status_code == 404


@pytest.mark.django_db
def test_rate_limit_429_after_30_requests():
    User.objects.create_user(username="ratelimit", email="rl@example.com", password="Password123")
    c = Client()
    c.login(username="ratelimit", password="Password123")
    for i in range(30):
        r = c.post(
            "/api/v1/check/sms/",
            data=json.dumps({"text": f"Привет, встреча в {i} часов"}),
            content_type="application/json",
        )
        assert r.status_code == 200
    r31 = c.post(
        "/api/v1/check/sms/",
        data=json.dumps({"text": "Привет, 31-й запрос"}),
        content_type="application/json",
    )
    assert r31.status_code == 429
    assert r31.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"


@pytest.mark.django_db
def test_model_unavailable_returns_503():
    User.objects.create_user(username="m503", email="m503@example.com", password="Password123")
    c = Client()
    c.login(username="m503", password="Password123")

    class FakePredictor:
        sms_available = False
        url_available = False

    with patch("apps.analysis.views.get_predictor", return_value=FakePredictor()):
        r1 = c.post("/api/v1/check/sms/", data=json.dumps({"text": "Проверка"}), content_type="application/json")
        r2 = c.post("/api/v1/check/url/", data=json.dumps({"url": "https://example.com"}), content_type="application/json")
    assert r1.status_code == 503
    assert r1.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert r2.status_code == 503


@pytest.mark.django_db
def test_privacy_store_sms_text_false_and_masking():
    u = User.objects.create_user(username="priv", email="priv@example.com", password="Password123")
    profile, _ = UserProfile.objects.get_or_create(user=u)
    profile.store_sms_text = False
    profile.save()

    c = Client()
    c.login(username="priv", password="Password123")
    secret_msg = "Переведите 50000 тенге на карту 4400123456789012 или звоните +7 701 123 45 67"
    r = c.post("/api/v1/check/sms/", data=json.dumps({"text": secret_msg}), content_type="application/json")
    assert r.status_code == 200
    chk = Check.objects.get(pk=r.json()["id"])
    assert chk.input_text is None
    assert "4400123456789012" not in chk.input_masked
    assert "701 123 45 67" not in chk.input_masked
    assert "***" in chk.input_masked


@pytest.mark.django_db
def test_admin_dashboard_hides_foreign_sms_text():
    u = User.objects.create_user(username="regular", email="reg@example.com", password="Password123")
    profile, _ = UserProfile.objects.get_or_create(user=u)
    profile.store_sms_text = True
    profile.save()
    Check.objects.create(
        user=u,
        check_type="sms",
        input_text="SUPER_SECRET_USER_TEXT_98765",
        input_masked="SUPER_SECRET_PREVIEW_98765",
        risk_score=80,
        risk_level="high",
        category="FRAUDULENT",
        recommendation="warn",
    )
    User.objects.create_superuser(username="admin_t", email="adm@example.com", password="Password123")
    c = Client()
    c.login(username="admin_t", password="Password123")
    resp = c.get("/dashboard/")
    assert resp.status_code == 200
    html = resp.content.decode("utf-8")
    assert "SUPER_SECRET_USER_TEXT_98765" not in html


@pytest.mark.django_db
def test_xss_escaping_and_sqli_protection():
    u = User.objects.create_user(username="secuser", email="sec@example.com", password="Password123")
    profile, _ = UserProfile.objects.get_or_create(user=u)
    profile.store_sms_text = True
    profile.save()

    c = Client()
    c.login(username="secuser", password="Password123")
    xss_payload = "<script>alert(1)</script> Срочно переведите деньги"
    r = c.post("/api/v1/check/sms/", data=json.dumps({"text": xss_payload}), content_type="application/json")
    assert r.status_code == 200
    chk_id = r.json()["id"]

    page = c.get(f"/checks/{chk_id}/").content.decode("utf-8")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page

    sqli = c.get("/history/?q=' OR 1=1 --")
    assert sqli.status_code == 200


@pytest.mark.django_db
def test_double_submit_deduplication_within_3s():
    User.objects.create_user(username="dedup", email="dd@example.com", password="Password123")
    c = Client()
    c.login(username="dedup", password="Password123")
    payload = json.dumps({"text": "Повторяющийся текст проверки для дедупликации"})
    r1 = c.post("/api/v1/check/sms/", data=payload, content_type="application/json")
    r2 = c.post("/api/v1/check/sms/", data=payload, content_type="application/json")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    assert Check.objects.count() == 1


@pytest.mark.django_db
def test_csv_export_and_profile_delete_all_history():
    u = User.objects.create_user(username="expuser", email="exp@example.com", password="Password123")
    c = Client()
    c.login(username="expuser", password="Password123")
    c.post("/api/v1/check/sms/", data=json.dumps({"text": "Привет, как дела?"}), content_type="application/json")
    assert Check.objects.filter(user=u).count() == 1

    csv_resp = c.get("/history/export/")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp["Content-Type"]
    assert "Категория" in csv_resp.content.decode("utf-8")

    prof_resp = c.post("/profile/", data={"action": "update_privacy", "store_sms_text": "on", "preferred_language": "kk"})
    assert prof_resp.status_code == 302
    u.profile.refresh_from_db()
    assert u.profile.store_sms_text is True
    assert u.profile.preferred_language == "kk"

    del_resp = c.post("/profile/", data={"action": "delete_all_history"})
    assert del_resp.status_code == 302
    assert Check.objects.filter(user=u).count() == 0


@pytest.mark.django_db
def test_health_and_stats_summary_endpoints():
    User.objects.create_user(username="stuser", email="st@example.com", password="Password123")
    c = Client()
    h = c.get("/api/v1/health/")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    c.login(username="stuser", password="Password123")
    st = c.get("/api/v1/stats/summary/?days=7")
    assert st.status_code == 200
    data = st.json()
    assert data["period_days"] == 7
    assert "total" in data
    assert "safe_count" in data
    assert c.get("/stats/?days=30").status_code == 200
