import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sleep_demo.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                occupation TEXT NOT NULL,
                sleep_duration REAL NOT NULL,
                physical_activity INTEGER NOT NULL,
                daily_steps INTEGER NOT NULL,
                predicted_score REAL NOT NULL,
                recommendations TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    *,
    age,
    gender,
    occupation,
    sleep_duration,
    physical_activity,
    daily_steps,
    predicted_score,
    recommendations,
):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, age, gender, occupation,
                sleep_duration, physical_activity, daily_steps,
                predicted_score, recommendations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                age,
                gender,
                occupation,
                sleep_duration,
                physical_activity,
                daily_steps,
                predicted_score,
                json.dumps(recommendations),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_predictions(limit=15):
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT id, created_at, age, gender, occupation,
                   sleep_duration, physical_activity, daily_steps,
                   predicted_score, recommendations
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "age": r["age"],
                    "gender": r["gender"],
                    "occupation": r["occupation"],
                    "sleep_duration": r["sleep_duration"],
                    "physical_activity": r["physical_activity"],
                    "daily_steps": r["daily_steps"],
                    "predicted_score": r["predicted_score"],
                    "recommendations": json.loads(r["recommendations"]),
                }
            )
        return rows
    finally:
        conn.close()
