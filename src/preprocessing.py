"""
Metin ön işleme modülü
- Küçük harfe çevirme
- Noktalama/özel karakter temizleme
- Stopword kaldırma
- Tokenization
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Gerekli NLTK verilerini indir (ilk çalıştırmada)
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Ham metni temizler: lowercase, URL/HTML/noktalama temizliği."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)            # HTML tagleri
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URL'ler
    text = re.sub(r"[^a-z\s]", " ", text)          # sadece harf bırak
    text = re.sub(r"\s+", " ", text).strip()       # fazla boşluk
    return text


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """Tokenize eder ve stopword'leri kaldırır."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess(text: str) -> str:
    """Tam ön işleme pipeline'ı: temizle -> tokenize -> stopword temizle -> birleştir."""
    cleaned = clean_text(text)
    tokens = tokenize_and_remove_stopwords(cleaned)
    return " ".join(tokens)


def parse_ft_line(line: str):
    """fastText formatlı satırı (__label__1 metin...) parse eder.
    Döner: (label, text) -> label: 0=negatif, 1=pozitif
    """
    line = line.strip()
    if line.startswith("__label__1"):
        label = 0
        text = line[len("__label__1"):].strip()
    elif line.startswith("__label__2"):
        label = 1
        text = line[len("__label__2"):].strip()
    else:
        return None, None
    return label, text


if __name__ == "__main__":
    sample = "This product is AMAZING!! I loved it <br> Check http://example.com for more."
    print("Orijinal :", sample)
    print("Temiz    :", clean_text(sample))
    print("İşlenmiş :", preprocess(sample))
