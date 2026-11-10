import re
import string
import unicodedata
import warnings
import pandas as pd
import os

TAGALOG_STOPWORDS = {
    'ang','mga','ng','sa','na','ay','at','ito','iyon','yan','yun','si','ni','kay',
    'ako','ikaw','ka','siya','kami','tayo','kayo','sila','ko','mo','niya','namin',
    'natin','ninyo','nila','akin','iyo','kanya','amin','atin','inyo','kanila',
    'iyan','doon','dito','diyan','paano','bakit','saan','kailan','sino',
    'ano','alin','ba','po','opo','oo','hindi','wag','huwag','lang','din','rin',
    'pa','nga','naman','kung','kapag','dahil','kasi','para','upang','pero',
    'ngunit','o','may','meron','wala','yung','yong','nung','noon',
    'ganito','ganyan','ganoon','sana','daw','raw','muna','ulit','uli','talaga'
}

prefix_map = {
    "0907": "Smart", "0908": "Smart", "0909": "Smart", "0910": "Smart", 
    "0911": "Smart", "0912": "Smart", "0913": "Smart", "0914": "Smart", 
    "0918": "Smart", "0919": "Smart", "0920": "Smart", "0921": "Smart", "0928": "Smart", 
    "0929": "Smart", "0930": "Smart", "0938": "Smart", "0939": "Smart", "0946": "Smart", 
    "0947": "Smart", "0948": "Smart", "0949": "Smart", "0950": "Smart", "0951": "Smart", 
    "0960": "Smart", "0961": "Smart", "0962": "Smart", "0963": "Smart", "0964": "Smart",
    "0968": "Smart", "0969": "Smart", "0970": "Smart", "0981": "Smart", "0985": "Smart", 
    "0989": "Smart", "0998": "Smart", "0999": "Smart", "0922": "Smart", "0923": "Smart",
    "0924": "Smart", "0925": "Smart", "0931": "Smart", "0932": "Smart",
    "0933": "Smart", "0934": "Smart", "0941": "Smart", "0942": "Smart",
    "0943": "Smart", "0944": "Smart", "0973": "Smart", "0974": "Smart",
    
    "0905": "Globe", "0906": "Globe", "0915": "Globe", "0916": "Globe", "0917": "Globe", 
    "0926": "Globe", "0927": "Globe", "0935": "Globe", "0936": "Globe", "0937": "Globe", 
    "0945": "Globe", "0953": "Globe", "0954": "Globe", "0955": "Globe", "0956": "Globe", 
    "0965": "Globe", "0966": "Globe", "0967": "Globe", "0975": "Globe", "0977": "Globe", 
    "0995": "Globe", "0997": "Globe", "0978": "Globe", "0979": "Globe",

    "0991": "DITO", "0992": "DITO", "0993": "DITO", "0994": "DITO",
}

def _load_english_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english')), 'nltk'
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS), 'sklearn-fallback'

_ENGLISH_STOPWORDS, STOPWORD_SOURCE = _load_english_stopwords()
STOPWORDS = _ENGLISH_STOPWORDS | TAGALOG_STOPWORDS

LEET_MAP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a'})
PLACEHOLDERS = ('urltoken', 'phonetoken', 'moneytoken')
TLDS = r'(?:com|click|icu|cyou|bond|shop|help|online|qpon|xyz|net|eu|ph|info|top|site|wang|at|live|work|fun|cf|tk|gq|ml|ga|co)'

URL_PROTO_RE = re.compile(r'(?:https?://|www\.)\S+', re.I)
URL_BARE_RE = re.compile(rf'\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.{TLDS}(?:\.ph)?(?:/\S*)?\b', re.I)
PHONE_RE = re.compile(r'(?<!\w)(?:\+?63|0)9\d{9}(?!\w)|(?<!\w)\d{10}(?!\w)')
MONEY_RE = re.compile(r'(php\s?\d[\d,\.]*|₱\s?\d[\d,\.]*|\bp\d+(?:[,\.]\d+)+\b|\bp\d{2,}[\d,]*\b|\d[\d,]*p\b|\d+k\b|\d+m\b)', re.I)
TIME_OR_ORDINAL_RE = re.compile(r'\d{1,2}(?:am|pm|st|nd|rd|th)', re.I)

OTP_PHRASES_RE = re.compile(r'\b(?:otp|pin|verification\s*code|one[\s\-]*time\s*password)\b', re.I)
URGENCY_RE = re.compile(r'\b(?:urgent|expire[ds]?|today|now|immediately|asap|last chance|warning|alert|hurry|claim\s*now|deactivated|disabled|suspended|valid for|limited time)\b', re.I)
LEETSPEAK_RE = re.compile(r'\b[a-zA-Z]+[043157@\$][a-zA-Z]+[a-zA-Z0-9]*\b|\b[043157@\$][a-zA-Z]+[a-zA-Z0-9]*\b', re.I)
TYPO_SQUATTING_RE = re.compile(r'\b(?:bd0|bpl|g[-_\.]*cash|gc4sh|sm4rt|gl0be|m4ya|g\s*cash|paym4ya|p4ymaya)\b', re.I)
URL_SHORTENER_RE = re.compile(r'\b(?:bit\.ly|cutt\.ly|tinyurl\.com|tiny\.cc|t\.co|goo\.gl|is\.gd|cli\.gs|ksk\.la|loom\.ly|dub\.sh|surl\.lu|tinyurl|onelink\.me)\b', re.I)

TARGET_MAP = {'Legitimate (ham)': 0, 'Smishing (scam)': 1, 'Legitimate': 0, 'Smishing': 1, '0': 0, '1': 1, 0: 0, 1: 1}
CARRIER_MAP = {
    "Smart/TNT": "Smart", "Smart/Sun": "Smart", "Sun": "Smart", "TNT": "Smart",
    "TM": "Globe", "Globe/TM": "Globe", "Globe PostPaid": "Globe",
    "Globe Postpaid": "Globe", "<unknown>": "Unknown", "NaN": "Unknown"
}

REQUIRED_COLUMNS = ['date', 'text', 'target']

def normalize_leet(word: str) -> str:
    if word in PLACEHOLDERS or re.fullmatch(r'\d+', word):
        return word
    if TIME_OR_ORDINAL_RE.fullmatch(word):
        return word
    if re.search(r'[a-z]', word) and re.search(r'[0134579@]', word):
        return word.replace('@', 'a').translate(LEET_MAP)
    return word

def transform_text(text: str) -> str:
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
    
    has_shortened = bool(URL_SHORTENER_RE.search(text))
    has_url = bool(URL_PROTO_RE.search(text) or URL_BARE_RE.search(text) or has_shortened)
    
    return {
        'has_url': int(has_url),
        'has_money_mention': int(bool(MONEY_RE.search(text))),
        'has_phone_in_text': int(bool(PHONE_RE.search(text))),
        'exclamation_count': text.count('!'),
        'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
        'has_otp': int(bool(OTP_PHRASES_RE.search(text))),
        'urgency': int(bool(URGENCY_RE.search(text))),
        'has_leetspeaks': int(bool(LEETSPEAK_RE.search(text))),
        'typo_squatting': int(bool(TYPO_SQUATTING_RE.search(text))),
        'has_url_shortened': int(has_shortened)
    }

def count_words(words: str) -> int:
    return len(str(words).split())

def update_carrier(row):
    sender = str(row.get('sender', ''))
    if not sender or sender == 'nan':
        return 'Unknown'
        
    match = re.search(r'^(\+?63|0)(9\d{2})', sender)
    if match:
        prefix = '0' + match.group(2)
        return prefix_map.get(prefix, 'Unknown Mobile')
        
    sender_lower = sender.lower()
    if 'smart' in sender_lower or 'tnt' in sender_lower:
        return 'Smart'
    elif 'globe' in sender_lower or 'tm' in sender_lower:
        return 'Globe'
    elif 'dito' in sender_lower:
        return 'DITO'
    elif 'gomo' in sender_lower:
        return 'GOMO'
    
    if re.search(r'[a-zA-Z]', sender):
        return 'Alphanumeric Sender'

    if sender.isdigit() and len(sender) <= 6:
        return 'Shortcode'
        
    return 'Unknown'

def transform(df: pd.DataFrame, drop_spam: bool = True,
              one_hot_carrier: bool = False, drop_empty: bool = True) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]
    
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise KeyError(f"Raw data is missing required columns: {missing}")

    data['target'] = data['target'].astype(str).str.strip()
    if drop_spam:
        data = data[data['target'].str.lower() != 'spam']
    
    data = data.dropna(subset=['text']).reset_index(drop=True)
    data['target'] = data['target'].map(lambda x: TARGET_MAP.get(x, x))
    data['target'] = pd.to_numeric(data['target'], errors='coerce')

    signals = pd.DataFrame(data['text'].map(extract_signal_feats).tolist(), index=data.index)
    data = pd.concat([data, signals], axis=1)
    
    if 'carrier' not in data.columns:
        data['carrier'] = 'Unknown'
    
    data['carrier'] = data['carrier'].astype(str).str.strip().replace(CARRIER_MAP)
    data.loc[data['carrier'] == 'nan', 'carrier'] = 'Unknown'
    
    if 'sender' in data.columns:
        data.loc[data['carrier'] == 'Unknown', 'carrier'] = data[data['carrier'] == 'Unknown'].apply(update_carrier, axis=1)
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month_name()

    data['text_raw'] = data['text'].map(lambda s: ' '.join(str(s).split()))
    data['punctuation_count'] = data['text_raw'].map(lambda s: sum(ch in string.punctuation for ch in s))
    data['word_count'] = data['text_raw'].map(count_words)
    data['text_count'] = data['text_raw'].map(len)
    data['text'] = data['text_raw'].map(preprocess_text)
    
    empty = data['text'].eq('')
    if empty.any() and drop_empty:
        data = data[~empty].reset_index(drop=True)
        
    data['dedup_group_id'] = pd.factorize(data['text'])[0]

    if one_hot_carrier and 'carrier' in data.columns:
        data = pd.get_dummies(data, columns=['carrier'], dtype=int)
        
    return data

def main(master_dataset_path: str, new_datasets_paths: list, output_path: str):

    """
    Main orchestration function to merge and preprocess a master dataset 
    with any number of new datasets containing standard columns.
    """

    print(f"Loading master dataset: {master_dataset_path}")
    try:
        df_master = pd.read_csv(master_dataset_path)
        # Revert to raw schema for clean reprocessing
        core_columns = ['date', 'sender', 'carrier', 'target', 'source', 'category']
        existing_cols = [c for c in core_columns if c in df_master.columns]
        df_master_raw = df_master[existing_cols].copy()
        if 'text_raw' in df_master.columns:
            df_master_raw['text'] = df_master['text_raw']
        elif 'text' in df_master.columns:
            df_master_raw['text'] = df_master['text']
        else:
            raise KeyError("Master dataset missing 'text' or 'text_raw' column.")
    except FileNotFoundError:
        print(f"Master file {master_dataset_path} not found. Starting fresh.")
        df_master_raw = pd.DataFrame()

    dfs_to_concat = [df_master_raw]

    for path in new_datasets_paths:
        print(f"Loading new batch: {path}")
        try:
            df_new = pd.read_csv(path)
            df_new.columns = [c.lower() for c in df_new.columns]
            dfs_to_concat.append(df_new)
        except FileNotFoundError:
            print(f"File {path} not found. Skipping.")

    if not dfs_to_concat:
        print("No datasets loaded. Exiting.")
        return

    if not df_master_raw.empty:
        df_master_raw.columns = [c.lower() for c in df_master_raw.columns]

    print("Concatenating all datasets...")
    df_combined_raw = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"Total raw rows combined: {len(df_combined_raw)}")

    print("Running preprocessing pipeline...")
    df_final = transform(df_combined_raw)

    print(f"Total finalized rows: {len(df_final)}")
    print(f"Final Features: {list(df_final.columns)}")
    
    df_final.to_csv(output_path, index=False)
    print(f"Successfully saved to: {output_path}")
    return df_final