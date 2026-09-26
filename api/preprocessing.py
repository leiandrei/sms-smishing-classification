import unicodedata
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

# global dictionary
TAGALOG_STOPWORDS = {
    'ang','mga','ng','sa','na','ay','at','ito','iyon','yan','yun','si','ni','kay',
    'ako','ikaw','ka','siya','kami','tayo','kayo','sila','ko','mo','niya','namin',
    'natin','ninyo','nila','akin','iyo','kanya','amin','atin','inyo','kanila',
    'iyan','doon','dito','diyan','paano','bakit','saan','kailan','sino',
    'ano','alin','ba','po','opo','oo','hindi','wag','huwag','lang','din','rin',
    'pa','nga','naman','kung','kapag','dahil','kasi','para','upang','pero',
    'ngunit','o','may','meron','wala','yung','yong','nung','noon',
    'ganito','ganyan','ganoon','sana','daw','raw','muna','ulit','uli','talaga'}

STOPWORDS = set(stopwords.words('english')) | TAGALOG_STOPWORDS
LEET_MAP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'})
TLDS = r'(?:com|click|icu|cyou|bond|shop|help|online|qpon|xyz|net|eu|ph|info|top|site|wang|at|live|work|fun|cf|tk|gq|ml|ga|co)'

def normalize_leet(word: str) -> str:
    if word in ('urltoken', 'phonetoken', 'moneytoken'):
        return word
    if re.fullmatch(r'\d+', word):
        return word
    if re.search(r'[a-z]', word) and re.search(r'[0134579@]', word):
        return word.replace('@', 'a').translate(LEET_MAP)
    return word

def transform_text(text: str):

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

    text = text.lower()
    tokens = [normalize_leet(t) for t in text.split()]
    text = ' '.join(tokens)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = [t for t in str(text).split() if t not in STOPWORDS and t != '']
    return ' '.join(tokens)

if __name__ == "__main__":
    print(transform_text("Test m3ssage w/ a URL http://test.com and p500"))