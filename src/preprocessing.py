import re
import string
import unicodedata
import warnings
import pandas as pd

TAGALOG_STOPWORDS = {
    'ang','mga','ng','sa','na','ay','at','ito','iyon','yan','yun','si','ni','kay',
    'ako','ikaw','ka','siya','kami','tayo','kayo','sila','ko','mo','niya','namin',
    'natin','ninyo','nila','akin','iyo','kanya','amin','atin','inyo','kanila',
    'iyan','doon','dito','diyan','paano','bakit','saan','kailan','sino',
    'ano','alin','ba','po','opo','oo','hindi','wag','huwag','lang','din','rin',
    'pa','nga','naman','kung','kapag','dahil','kasi','para','upang','pero',
    'ngunit','o','may','meron','wala','yung','yong','nung','noon',
    'ganito','ganyan','ganoon','sana','daw','raw','muna','ulit','uli','talaga'}


def _load_english_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english')), 'nltk'
    except Exception:
        pass
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words('english')), 'nltk'
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        warnings.warn(
            "NLTK stopwords unavailable -> using scikit-learn's English list. "
            "This differs from NLTK's list; use the same source for training "
            "and serving.", stacklevel=2)
        return set(ENGLISH_STOP_WORDS), 'sklearn-fallback'


_ENGLISH_STOPWORDS, STOPWORD_SOURCE = _load_english_stopwords()
STOPWORDS = _ENGLISH_STOPWORDS | TAGALOG_STOPWORDS

LEET_MAP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'})
PLACEHOLDERS = ('urltoken', 'phonetoken', 'moneytoken')

TLDS = r'(?:com|click|icu|cyou|bond|shop|help|online|qpon|xyz|net|eu|ph|info|top|site|wang|at|live|work|fun|cf|tk|gq|ml|ga|co)'

URL_PROTO_RE = re.compile(r'(?:https?://|www\.)\S+', re.I)
URL_BARE_RE = re.compile(
    rf'\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.{TLDS}(?:\.ph)?(?:/\S*)?\b', re.I)


PHONE_RE = re.compile(r'(?<!\w)(?:\+?63|0)9\d{9}(?!\w)|(?<!\w)\d{10}(?!\w)')

MONEY_RE = re.compile(
    r'(php\s?\d[\d,\.]*|₱\s?\d[\d,\.]*|\bp\d+(?:[,\.]\d+)+\b|'
    r'\bp\d{2,}[\d,]*\b|\d[\d,]*p\b|\d+k\b|\d+m\b)', re.I)

TIME_OR_ORDINAL_RE = re.compile(r'\d{1,2}(?:am|pm|st|nd|rd|th)')


TARGET_MAP = {'Legitimate (ham)': 0, 'Smishing (scam)': 1}
TARGET_NAMES = ['Legitimate', 'Smishing']
TARGET_MAP_3CLASS = {'Legitimate (ham)': 0, 'Smishing (scam)': 1, 'Spam': 2}
TARGET_NAMES_3CLASS = ['Legitimate', 'Smishing', 'Spam']

CARRIER_MAP = {
    "Smart/TNT": "Smart", "Smart/Sun": "Smart", "Sun": "Smart", "TNT": "Smart",
    "TM": "Globe", "Globe/TM": "Globe", "Globe PostPaid": "Globe",
    "Globe Postpaid": "Globe", "<unknown>": "Unknown",
}

REQUIRED_COLUMNS = ['date', 'text', 'Target']
PHONE_COLUMNS = ['masked_celphone_number', 'hashed_cellphone_number']


def normalize_leet(word: str) -> str:
    if word in PLACEHOLDERS or re.fullmatch(r'\d+', word):
        return word
    if TIME_OR_ORDINAL_RE.fullmatch(word):
        return word
    if re.search(r'[a-z]', word) and re.search(r'[0134579@]', word):
        return word.replace('@', 'a').translate(LEET_MAP)
    return word


def transform_text(text: str) -> str:
    if isinstance(text, (pd.Series, pd.DataFrame)):
        raise TypeError("transform_text() takes ONE string; for a column use "
                        "series.map(transform_text).")
    if text is None or (isinstance(text, float) and text != text):
        return ''

    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = ' '.join(text.split())

    text = URL_PROTO_RE.sub(' urltoken ', text)
    text = URL_BARE_RE.sub(' urltoken ', text)
    text = PHONE_RE.sub(' phonetoken ', text)
    text = MONEY_RE.sub(' moneytoken ', text)

    text = ' '.join(normalize_leet(t) for t in re.split(r'[^a-z0-9@]+', text) if t)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ''
    return ' '.join(t for t in str(text).split() if t not in STOPWORDS)


def preprocess_text(raw_text: str) -> str:
    return remove_stopwords(transform_text(raw_text))


def extract_signal_feats(raw_text: str) -> dict:
    text = '' if raw_text is None else str(raw_text)
    return {
        'has_url': int(bool(URL_PROTO_RE.search(text) or URL_BARE_RE.search(text))),
        'has_money_mention': int(bool(MONEY_RE.search(text))),
        'has_phone_in_text': int(bool(PHONE_RE.search(text))),
        'exclamation_count': text.count('!'),
        'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
    }


def count_words(words: str) -> int:
    return len(str(words).split())

def transform(df: pd.DataFrame, drop_spam: bool = True,
              one_hot_carrier: bool = False, drop_empty: bool = True) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Raw data is missing required columns: {missing}")

    data = df.copy()
    data['Target'] = data['Target'].astype(str).str.strip()
    if drop_spam:
        data = data[data['Target'] != 'Spam']
    target_map = TARGET_MAP if drop_spam else TARGET_MAP_3CLASS
    unknown = sorted(set(data['Target']) - set(target_map))
    if unknown:
        raise ValueError(f"Unexpected Target labels {unknown}; "
                         f"expected {sorted(target_map)}.")
    data = data.dropna(subset=['text']).reset_index(drop=True)
    data['Target'] = data['Target'].map(target_map)

    signals = pd.DataFrame(data['text'].map(extract_signal_feats).tolist(),
                           index=data.index)
    data = pd.concat([data, signals], axis=1)
    data = data.drop(columns=[c for c in PHONE_COLUMNS if c in data.columns])
    if 'carrier' in data.columns:
        data['carrier'] = data['carrier'].astype(str).str.strip().replace(CARRIER_MAP)
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month_name()

    data['text_raw'] = data['text'].map(lambda s: ' '.join(str(s).split()))
    data['punctuation_count'] = data['text_raw'].map(
        lambda s: sum(ch in string.punctuation for ch in s))
    data['word_count'] = data['text_raw'].map(count_words)
    data['text_count'] = data['text_raw'].map(len)
    data['text'] = data['text_raw'].map(preprocess_text)
    empty = data['text'].eq('')
    if empty.any():
        warnings.warn(f"{int(empty.sum())} message(s) have no text left after "
                      f"cleaning" + (" and were dropped." if drop_empty else "."),
                      stacklevel=2)
        if drop_empty:
            data = data[~empty].reset_index(drop=True)
    data['dedup_group_id'] = pd.factorize(data['text'])[0]

    data.columns = data.columns.str.lower()
    if one_hot_carrier and 'carrier' in data.columns:
        data = pd.get_dummies(data, columns=['carrier'], dtype=int)
    return data
