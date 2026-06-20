"""
SQLite veritabanı modülü.
Tablo: predictions (yorum metni, tahmin edilen duygu, güven skoru, zaman damgası)
"""
import os
import sqlite3
from datetime import datetime

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "..", "data", "sentiment.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence_score REAL,
            model_used TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(review_text: str, predicted_sentiment: str,
                       confidence_score: float, model_used: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions (review_text, predicted_sentiment, confidence_score, model_used, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (review_text, predicted_sentiment, confidence_score, model_used,
         datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def fetch_all_predictions():
    conn = get_connection()
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC", conn)
    conn.close()
    return df


def fetch_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT predicted_sentiment, COUNT(*) FROM predictions GROUP BY predicted_sentiment")
    rows = cur.fetchall()
    conn.close()
    return dict(rows)


if __name__ == "__main__":
    init_db()
    print(f"Veritabanı oluşturuldu: {DB_PATH}")
