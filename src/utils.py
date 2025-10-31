import re
import nltk
from nltk.corpus import stopwords

_NLTK_READY = False


def ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    _NLTK_READY = True


def preprocess_text(text: str) -> str:
    """Basic text cleaning: lower, remove non-word chars and extra spaces.

    Uses NLTK for tokenization and stopword removal (English).
    """
    if not isinstance(text, str):
        text = str(text)
    ensure_nltk()
    text = text.lower()
    # remove URLs and user handles
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    # keep words and simple punctuation removal
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return ' '.join(tokens)
