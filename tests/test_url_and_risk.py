from ml.url_features import extract_url_features, validate_url
from ml.rules import analyze_text_rules, RULE_GROUPS
from ml.risk import calculate_risk
from ml.analyzer import extract_urls, detect_language

def test_url_features_and_validation():
    assert validate_url("https://example.com/login")
    assert not validate_url("not a url")
    features = extract_url_features("http://192.0.2.1/login?next=x")
    assert features["has_ip"] == 1 and features["has_login_word"] == 1

def test_url_extraction_max_three_and_schemeless():
    found = extract_urls("visit https://a.com and www.b.kz or bit.ly/x then https://d.org")
    assert len(found) == 3 and found[0].startswith("https://")

def test_rules_have_required_groups_and_multilingual_phrases():
    assert len(RULE_GROUPS) >= 6
    assert all(len(v) >= 10 for v in RULE_GROUPS.values())
    assert analyze_text_rules("Срочно переведите деньги и сообщите код")["group_count"] >= 3
    assert analyze_text_rules("Ақша аударыңыз, шұғыл")["group_count"] >= 2
    assert analyze_text_rules("urgent send money now")["group_count"] >= 2

def test_risk_is_bounded_and_monotonic():
    low = calculate_risk(0.1, 0.1, 0)
    high = calculate_risk(0.9, 0.9, 5)
    assert 0 <= low["risk_score"] <= 100
    assert high["risk_score"] > low["risk_score"]
    assert high["risk_level"] == "high"

def test_language_detection():
    assert detect_language("Сәлем, ақша") == "kk"
    assert detect_language("Здравствуйте") == "ru"
    assert detect_language("Hello there") == "en"


def test_required_demo_scenarios():
    from ml.analyzer import analyze_message
    safe = analyze_message("Hello, see you tomorrow at lunch.")
    transfer = analyze_message("Срочно переведите деньги на безопасный счёт и сообщите код.")
    suspicious = analyze_message("Проверьте аккаунт: https://secure-login-example.xyz/verify")
    url_only = analyze_message("https://example.com/login")
    no_url_social = analyze_message("Ваш счёт заблокирован. Срочно сообщите код из SMS.")
    assert safe["urls"] == [] and safe["p_url_max"] is None
    assert transfer["risk_score"] > safe["risk_score"]
    assert suspicious["urls"] and suspicious["p_url_max"] is not None
    assert url_only["check_type"] == "url" and url_only["p_sms"] is None
    assert no_url_social["urls"] == [] and no_url_social["p_url_max"] is None
    assert no_url_social["risk_score"] > safe["risk_score"]

def test_joint_analyzer_handles_one_and_three_urls():
    from ml.analyzer import analyze_message
    one = analyze_message("Проверьте https://a.example.com/login")
    three = analyze_message("https://a.com https://b.kz https://c.org https://d.net")
    assert len(one["urls"]) == 1
    assert len(three["urls"]) == 3


def test_url_features_extended_and_reason():
    from ml.url_features import validate_url_with_reason, explain_url
    ok, _ = validate_url_with_reason("https://example.com/path?a=1&b=2")
    assert ok
    bad, reason = validate_url_with_reason("javascript:alert(1)")
    assert not bad and reason

    puny = extract_url_features("http://xn--e1afmkfd.xn--p1ai/login")
    assert puny["has_punycode"] == 1
    spoof = extract_url_features("http://kaspi1.kz/auth")
    assert spoof["brand_spoof"] == 1
    short = extract_url_features("https://bit.ly/3abc")
    assert short["has_shortener"] == 1
    assert len(explain_url(spoof)) >= 1


def test_risk_thresholds_and_overrides():
    from ml.risk import compute_risk
    # Без URL веса (0.5, 0.2) нормируются на 0.7: 0.47 * (5/7) * 100 ≈ 34, 0.49 * (5/7) * 100 = 35
    r34 = compute_risk(p_sms=0.47, se_groups=[])
    assert r34.risk_score == 34 and r34.category == "SAFE"
    r35 = compute_risk(p_sms=0.49, se_groups=[])
    assert r35.risk_score == 35 and r35.category == "SUSPICIOUS"
    r_code = compute_risk(p_sms=0.1, se_groups=["CODE_REQUEST"])
    assert r_code.category in ("SUSPICIOUS", "FRAUDULENT")
    assert r_code.risk_score >= 35
    r_high = compute_risk(p_sms=0.9, p_url_max=0.9, se_groups=["URGENCY", "CODE_REQUEST"])
    assert r_high.risk_score >= 70 and r_high.category == "FRAUDULENT"
