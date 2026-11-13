"""Словарь правил социальной инженерии (ФТ-2.5, Задание 12 п.4):
8 групп, не менее 10 выражений на каждом языке (русский, казахский, английский) в каждой группе.
"""
import re

RULE_GROUPS_LANG = {
    "URGENCY": {
        "title": "Срочность и давление временем",
        "weight": 15,
        "ru": [
            "срочно", "немедленно", "в течение 24 часов", "последний шанс", "действуйте сейчас",
            "без промедления", "прямо сейчас", "до конца дня", "время ограничено", "не откладывайте",
            "экстренное сообщение", "требуется срочное",
        ],
        "kk": [
            "шұғыл", "дереу", "бүгін аяқталады", "тез арада", "уақыт шектеулі",
            "дәл қазір", "кешіктірмей", "соңғы мүмкіндік", "жедел түрде", "шұғыл хабарлама",
            "уақыт қалмады", "тезірек",
        ],
        "en": [
            "immediately", "urgent", "act now", "within 24 hours", "last chance",
            "right now", "time is running out", "do not delay", "expires today", "urgent action",
            "final notice", "asap",
        ],
    },
    "MONEY_TRANSFER": {
        "title": "Требование перевода денег или оплаты",
        "weight": 20,
        "ru": [
            "переведите деньги", "сделайте перевод", "оплатите комиссию", "верните средства", "переведи",
            "оплатите сбор", "переведите средства", "оплатите пошлину", "на этот номер", "на безопасный счёт",
            "резервный счёт", "погасите задолженность",
        ],
        "kk": [
            "аударым жасаңыз", "ақша аударыңыз", "төлем жасаңыз", "комиссия төлеңіз", "қаражатты аударыңыз",
            "қауіпсіз шотқа", "резервтік шотқа", "кедендік комиссия", "жеткізу ақысын", "қарызды төлеңіз",
            "шотқа салыңыз", "теңге аударыңыз",
        ],
        "en": [
            "send money", "wire transfer", "pay the fee", "transfer funds", "make a payment",
            "pay customs fee", "safe account", "processing fee", "pay immediately", "send cash",
            "payment required", "settle your debt",
        ],
    },
    "BANK_CARD": {
        "title": "Запрос реквизитов банковской карты",
        "weight": 20,
        "ru": [
            "данные карты", "номер карты", "cvv", "cvc", "срок действия карты",
            "банковской карты", "реквизиты карты", "пин-код", "зарплатная карта", "введите карту",
            "обратной стороны карты", "подтвердите карту",
        ],
        "kk": [
            "карта нөмірі", "карта деректері", "банк картасы", "картаның мерзімі", "пин-кодты",
            "картаңыздың деректерін", "картаны растаңыз", "банк шоты", "карта реквизиттері", "карта кодтарын",
            "төлем картасы", "карта нөмірін",
        ],
        "en": [
            "card details", "card number", "security code", "bank account", "expiration date",
            "credit card", "debit card", "enter your card", "card verification", "pin code",
            "billing information", "card holder",
        ],
    },
    "ACCOUNT_BLOCK": {
        "title": "Угроза блокировки карты или аккаунта",
        "weight": 20,
        "ru": [
            "аккаунт заблокирован", "счёт заблокирован", "карта заблокирована", "подозрительный вход", "заморожен",
            "доступ ограничен", "будет заблокирован", "приостановлено", "учётная запись заблокирована", "взломан",
            "попытка входа", "блокировки транзакции",
        ],
        "kk": [
            "шотыңыз бұғатталды", "картаңыз бұғатталды", "аккаунт бұғатталды", "кіру шектелді", "бөтен кіру",
            "картасы шектелді", "уақытша тоқтатылды", "бұғатталады", "күдікті операция", "күдікті аударым",
            "жеке кабинет бұғатталды", "шот тоқтатылды",
        ],
        "en": [
            "account suspended", "account blocked", "unusual login", "account locked", "card blocked",
            "will be suspended", "unauthorized access", "security alert", "temporarily frozen", "access restricted",
            "suspicious activity", "compromised account",
        ],
    },
    "PRIZE": {
        "title": "Обещание выигрыша, приза или бонуса",
        "weight": 15,
        "ru": [
            "вы выиграли", "получите приз", "вам начислен бонус", "вы стали победителем", "розыгрыш",
            "заберите выигрыш", "призовой счёт", "победителем лотереи", "бесплатный подарок", "компенсация",
            "социальная выплата", "инвестиционный бонус",
        ],
        "kk": [
            "сіз ұтып алдыңыз", "сыйлық ұттыңыз", "жүлде алыңыз", "ұтыс жеңімпазы", "бонус берілді",
            "лотереядан", "жүлде қорынан", "тегін сыйлық", "мемлекеттік жәрдемақы", "ұтысты алу",
            "жеңімпаз атандыңыз", "арнайы сыйлық",
        ],
        "en": [
            "you won", "claim your prize", "congratulations winner", "claim reward", "gift card",
            "free gift", "lottery winner", "cash prize", "bonus credited", "exclusive reward",
            "selected as a winner", "won a prize",
        ],
    },
    "CODE_REQUEST": {
        "title": "Выманивание кода из SMS или пароля",
        "weight": 30,
        "ru": [
            "сообщите код", "назовите код", "код из sms", "одноразовый пароль", "назовите одноразовый",
            "код отмены", "введите пароль", "смените пароль", "передайте код", "скажите код",
            "продиктуйте код", "пароль из сообщения",
        ],
        "kk": [
            "растау кодын жіберіңіз", "кодты айтыңыз", "sms кодын беріңіз", "құпиясөзді жіберіңіз", "бірреттік код",
            "құпиясөзді енгізіңіз", "кодты хабарлаңыз", "растау коды", "sms-кодты", "кодты жазыңыз",
            "құпия кодты", "парольді жіберіңіз",
        ],
        "en": [
            "share the code", "send otp", "verification code", "one-time password", "enter your password",
            "confirm your password", "security code from sms", "tell the code", "provide the otp", "sms code",
            "authentication code", "passcode",
        ],
    },
    "AUTHORITY": {
        "title": "Маскировка под банк, госорган или службу",
        "weight": 15,
        "ru": [
            "служба безопасности", "полиция", "налоговая", "нацбанк", "сотрудник банка",
            "следователю", "служба доставки", "таможенного сбора", "государства", "судебный исполнитель",
            "прокуратура", "финансовый мониторинг",
        ],
        "kk": [
            "қауіпсіздік қызметі", "полициядан", "салық комитеті", "ұлттық банк", "банк қызметкері",
            "полицияға", "мемлекеттік орган", "кеден қызметі", "тергеуші", "сот орындаушысы",
            "прокуратура", "қаржы мониторингі",
        ],
        "en": [
            "security department", "law enforcement", "tax authority", "central bank", "bank officer",
            "police department", "customs service", "government agency", "federal reserve", "fraud department",
            "official notice", "court order",
        ],
    },
    "SECRECY": {
        "title": "Требование секретности и неразглашения",
        "weight": 15,
        "ru": [
            "никому не говори", "никому не говорите", "не сообщайте о звонке", "строго конфиденциально", "в тайне",
            "не рассказывайте", "между нами", "потом всё объясню", "без лишних слов", "не звоните никому",
            "секретная операция", "не разглашайте",
        ],
        "kk": [
            "ешкімге айтпаңыз", "құпия сақтаңыз", "ешкімге тіс жармаңыз", "тек біздің арамызда", "кейін түсіндіремін",
            "ешкімге хабарламаңыз", "жасырын түрде", "құпия операция", "бөтен адамға айтпа", "сыр сақтаңыз",
            "ешкім білмесін", "жария етпеңіз",
        ],
        "en": [
            "tell no one", "keep it secret", "strictly confidential", "do not tell anyone", "keep this private",
            "between us", "explain later", "do not share this", "secret operation", "do not contact anyone",
            "confidential matter", "keep quiet",
        ],
    },
}

RULE_GROUPS = {
    group: info["ru"] + info["kk"] + info["en"]
    for group, info in RULE_GROUPS_LANG.items()
}

GROUP_TITLES = {group: info["title"] for group, info in RULE_GROUPS_LANG.items()}
GROUP_WEIGHTS = {group: info["weight"] for group, info in RULE_GROUPS_LANG.items()}

_COMPILED = {
    k: [(term, re.compile(re.escape(term), re.I)) for term in terms]
    for k, terms in RULE_GROUPS.items()
}


def analyze_text_rules(text):
    value = str(text or "")
    hits = {}
    details = []
    for group, patterns in _COMPILED.items():
        found = [term for term, p in patterns if p.search(value)]
        if found:
            hits[group] = found
            details.append({
                "code": group,
                "title": GROUP_TITLES[group],
                "weight": GROUP_WEIGHTS[group],
                "phrases": found,
            })
    return {
        "groups": hits,
        "group_count": len(hits),
        "matched_phrases": [x for values in hits.values() for x in values],
        "details": details,
    }
