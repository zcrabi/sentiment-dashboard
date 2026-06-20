"""
test.ft.txt dosyasından dengeli bir örnek alır, ön işler ve CSV olarak kaydeder.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
import random
import pandas as pd
from preprocessing import parse_ft_line, preprocess

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.ft.txt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reviews_clean.csv")

SAMPLE_PER_CLASS = 15000  # her sınıftan 15K -> toplam 30K satır (proje için yeterli, hızlı)
random.seed(42)


def main():
    pos, neg = [], []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            label, text = parse_ft_line(line)
            if label is None:
                continue
            if label == 1 and len(pos) < SAMPLE_PER_CLASS:
                pos.append(text)
            elif label == 0 and len(neg) < SAMPLE_PER_CLASS:
                neg.append(text)
            if len(pos) >= SAMPLE_PER_CLASS and len(neg) >= SAMPLE_PER_CLASS:
                break

    print(f"Toplanan: {len(pos)} pozitif, {len(neg)} negatif")

    rows = [{"raw_text": t, "label": 1} for t in pos] + \
           [{"raw_text": t, "label": 0} for t in neg]
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Ön işleme uygulanıyor...")
    df["clean_text"] = df["raw_text"].apply(preprocess)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Kaydedildi: {OUT_PATH} ({len(df)} satır)")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
