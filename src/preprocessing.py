import unicodedata
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

# global dictionary for stopwords: tagalog
TAGALOG_STOPWORDS = {
    'ang','mga','ng','sa','na','ay','at','ito','iyon','yan','yun','si','ni','kay',
    'ako','ikaw','ka','siya','kami','tayo','kayo','sila','ko','mo','niya','namin',
    'natin','ninyo','nila','akin','iyo','kanya','amin','atin','inyo','kanila',
    'iyan','doon','dito','diyan','paano','bakit','saan','kailan','sino',
    'ano','alin','ba','po','opo','oo','hindi','wag','huwag','lang','din','rin',
    'pa','nga','naman','kung','kapag','dahil','kasi','para','upang','pero',
    'ngunit','o','may','meron','wala','yung','yong','nung','noon',
    'ganito','ganyan','ganoon','sana','daw','raw','muna','ulit','uli','talaga'}

# sets if it spots an english or tagalog stopwords
STOPWORDS = set(stopwords.words('english')) | TAGALOG_STOPWORDS

# leetspeaks mapping: jejemon wordings usually: e.g: 4k0 -> ako
LEET_MAP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'})

# tlds: top level domains: extracted url domains from the sms text messages
TLDS = r'(?:com|click|icu|cyou|bond|shop|help|online|qpon|xyz|net|eu|ph|info|top|site|wang|at|live|work|fun|cf|tk|gq|ml|ga|co)'

def normalize_leet(word: str) -> str:
    if word in ('urltoken', 'phonetoken', 'moneytoken'):
        return word
    if re.fullmatch(r'\d+', word):
        return word
    if re.search(r'[a-z]', word) and re.search(r'[0134579@]', word):
        return word.replace('@', 'a').translate(LEET_MAP)
    return word

def transform_text(text: str) -> str:
    text = str(text).lower()
    text = ' '.join(text.split())
    text = unicodedata.normalize('NFKC', text)

    text = re.sub(r'(https?://\S+|www\.\S+)', ' urltoken ', text, flags=re.I)
    text = re.sub(rf'\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.{TLDS}(?:\.ph)?(?:/\S*)?\b', ' urltoken ', text, flags=re.I)
    text = re.sub(r'\b09\d{9}\b|\b\d{10}\b', ' phonetoken ', text)
    text = re.sub(
        r'(php\s?\d[\d,\.]*|₱\s?\d[\d,\.]*|\bp\d+(?:[,\.]\d+)+\b|\bp\d{2,}[\d,]*\b|\d[\d,]*p\b|\d+k\b|\d+m\b)',
        ' moneytoken ', text, flags=re.I
    )

    tokens = [normalize_leet(t) for t in text.split()]
    text = ' '.join(tokens)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = [t for t in str(text).split() if t not in STOPWORDS and t != '']
    return ' '.join(tokens)

def extract_signal_feats(raw_text: str) -> pd.Series:
    text = str(raw_text)
    return pd.Series({
        'has_url': int(bool(re.search(rf'(https?://\S+|www\.\S+|\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.{TLDS}\b)', text, re.I))),
        'has_money_mention': int(bool(re.search(r'(php\s?\d|₱\s?\d|\bp\d+(?:[,\.]\d+)+\b|\bp\d{2,}[\d,]*\b|\d[\d,]*p\b|\d+k\b|\d+m\b)', text, re.I))),
        'has_phone_in_text': int(bool(re.search(r'\b09\d{9}\b|\b\+639\d{9}\b|\b\d{10}\b', text))),
        'exclamation_count': text.count('!'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    })

def transform(df: pd.DataFrame, encoder: LabelEncoder = None, is_train: bool = True) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The DataFrame is Empty.")

    data = df.copy()

    try:
        data = data.drop(['masked_celphone_number', 'hashed_cellphone_number'], axis=1, errors='ignore')
        data['date'] = pd.to_datetime(data['date'])
        
        carrier_map = {
            "Smart/TNT" : "Smart", "Smart/Sun" : "Smart", "Sun" : "Smart",
            "TNT" : "Smart", "TM" : "Globe", "Globe/TM" : "Globe",
            "Globe PostPaid" : "Globe", "<unknown>" : "Unknown"
        }
        data['carrier'] = data['carrier'].replace(carrier_map)

        data['text'] = data['text'].fillna('')

        data['punctuation_count'] = data['text'].apply(lambda x: sum(1 for char in x if char in string.punctuation))
        data['text_count'] = data['text'].apply(len)
        signal_feats = data['text'].apply(extract_signal_feats)
        data = pd.concat([data, signal_feats], axis=1)

        data['text'] = data['text'].apply(transform_text)
        data['text'] = data['text'].apply(remove_stopwords)
        data['word_count'] = data['text'].apply(lambda x: len(x.split()))

        if 'Target' in data.columns and encoder is not None:
            if is_train:
                data['Target'] = encoder.fit_transform(data["Target"])
            else:
                data['Target'] = encoder.transform(data["Target"])

        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month_name()
        
        data.columns = data.columns.str.lower()
        return data

    except Exception as e:
        raise Exception(f"Unable to transform the DataFrame: {e}") from e