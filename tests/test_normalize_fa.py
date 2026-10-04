from mybrain.ingest.normalize_fa import (
    detect_lang,
    has_persian,
    lexical_form,
    normalize_name,
    normalize_persian,
)


def test_arabic_letters_become_persian():
    assert normalize_persian("كتاب علي") == "کتاب علی"


def test_digits_and_tatweel_and_diacritics():
    assert normalize_persian("سال ۱۴۰۲ و ٢٠٢٤ مـــدیر مَدیر") == "سال 1402 و 2024 مدیر مدیر"


def test_zwnj_kept_in_normalized_but_space_in_lexical():
    s = "می‌روم"
    assert "‌" in normalize_persian(s)
    assert lexical_form(s) == "می روم"


def test_lexical_form_strips_punctuation_and_case():
    assert lexical_form("Hello, World! (Pars-Holding)؟") == "hello world pars holding"


def test_normalize_name_drops_honorifics_and_suffixes():
    assert normalize_name("آقای دکتر علی رضایی") == "علی رضایی"
    assert normalize_name("Dr. Ali Rezaei") == "ali rezaei"
    assert normalize_name("Pars Holding Co.") == "pars holding"


def test_detect_lang():
    assert detect_lang("این یک متن فارسی است") == "fa"
    assert detect_lang("This is English text") == "en"
    assert has_persian("mixed متن")
    assert not has_persian("plain")
