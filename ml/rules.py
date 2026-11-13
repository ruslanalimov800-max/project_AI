"""Правила социальной инженерии: русские, казахские и английские выражения."""
import re

RULE_GROUPS = {
 "URGENCY": ["срочно", "немедленно", "в течение 24 часов", "последний шанс", "действуйте сейчас", "шұғыл", "дереу", "бүгін аяқталады", "тез арада", "immediately", "urgent", "act now"],
 "MONEY_TRANSFER": ["переведите деньги", "сделайте перевод", "оплатите комиссию", "верните средства", "аударым жасаңыз", "ақша аударыңыз", "төлем жасаңыз", "комиссия төлеңіз", "send money", "wire transfer", "pay the fee", "transfer funds"],
 "BANK_CARD": ["данные карты", "номер карты", "cvv", "cvc", "срок действия карты", "карта нөмірі", "карта деректері", "банк картасы", "card details", "card number", "security code", "bank account"],
 "ACCOUNT_BLOCK": ["аккаунт заблокирован", "счёт заблокирован", "карта заблокирована", "подозрительный вход", "шотыңыз бұғатталды", "картаңыз бұғатталды", "аккаунт бұғатталды", "кіру шектелді", "account suspended", "account blocked", "unusual login", "account locked"],
 "PRIZE": ["вы выиграли", "получите приз", "вам начислен бонус", "вы стали победителем", "сіз ұтып алдыңыз", "сыйлық ұттыңыз", "жүлде алыңыз", "ұтыс жеңімпазы", "you won", "claim your prize", "congratulations winner", "claim reward"],
 "CODE_REQUEST": ["сообщите код", "назовите код", "код из sms", "одноразовый пароль", "растау кодын жіберіңіз", "кодты айтыңыз", "sms кодын беріңіз", "құпиясөзді жіберіңіз", "share the code", "send otp", "verification code", "one-time password"],
 "AUTHORITY": ["служба безопасности", "полиция", "налоговая", "нацбанк", "қауіпсіздік қызметі", "полициядан", "салық комитеті", "ұлттық банк", "security department", "law enforcement", "tax authority", "central bank"],
}
_COMPILED = {k: [re.compile(re.escape(x), re.I) for x in terms] for k, terms in RULE_GROUPS.items()}

def analyze_text_rules(text):
    value = str(text or "")
    hits = {}
    for group, patterns in _COMPILED.items():
        found = [p.pattern for p in patterns if p.search(value)]
        if found:
            hits[group] = found
    return {"groups": hits, "group_count": len(hits), "matched_phrases": [x for values in hits.values() for x in values]}
