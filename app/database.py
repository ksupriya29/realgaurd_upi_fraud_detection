"""
SQLite database layer for RealGuard transaction history.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "realguard.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the transactions table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id          TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            amount          REAL,
            avg_amount      REAL,
            merchant        TEXT,
            txn_type        TEXT,
            txn_hour        INTEGER,
            day_of_week     INTEGER,
            unusual_loc     INTEGER,
            unusual_amt     INTEGER,
            new_device      INTEGER,
            failed_attempts INTEGER,
            fraud_prob      REAL,
            risk_score      INTEGER,
            risk_level      TEXT,
            is_fraud        INTEGER
        )
    """)
    conn.commit()
    conn.close()


def insert_transaction(record: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO transactions
            (txn_id, created_at, amount, avg_amount, merchant, txn_type,
             txn_hour, day_of_week, unusual_loc, unusual_amt, new_device,
             failed_attempts, fraud_prob, risk_score, risk_level, is_fraud)
        VALUES
            (:txn_id, :created_at, :amount, :avg_amount, :merchant, :txn_type,
             :txn_hour, :day_of_week, :unusual_loc, :unusual_amt, :new_device,
             :failed_attempts, :fraud_prob, :risk_score, :risk_level, :is_fraud)
    """, record)
    conn.commit()
    conn.close()


def fetch_history(limit: int = 100, fraud_only: bool = False,
                  legit_only: bool = False, search: str = ""):
    conn = get_connection()
    where = []
    params = []

    if fraud_only:
        where.append("is_fraud = 1")
    elif legit_only:
        where.append("is_fraud = 0")

    if search:
        where.append("(txn_id LIKE ? OR merchant LIKE ? OR txn_type LIKE ?)")
        s = f"%{search}%"
        params.extend([s, s, s])

    sql = "SELECT * FROM transactions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


def fetch_analytics():
    """Return aggregated stats for the dashboard."""
    conn = get_connection()

    total     = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud_cnt = conn.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
    legit_cnt = total - fraud_cnt
    avg_amt   = conn.execute("SELECT AVG(amount) FROM transactions").fetchone()[0] or 0
    avg_risk  = conn.execute("SELECT AVG(risk_score) FROM transactions").fetchone()[0] or 0
    fraud_rate = round(fraud_cnt / total * 100, 2) if total else 0

    # By merchant
    by_merchant = conn.execute("""
        SELECT merchant,
               SUM(is_fraud) as fraud,
               COUNT(*) as total
        FROM transactions GROUP BY merchant
    """).fetchall()

    # By txn type
    by_type = conn.execute("""
        SELECT txn_type,
               SUM(is_fraud) as fraud,
               COUNT(*) as total
        FROM transactions GROUP BY txn_type
    """).fetchall()

    # By hour
    by_hour = conn.execute("""
        SELECT txn_hour,
               SUM(is_fraud) as fraud,
               COUNT(*) as total
        FROM transactions GROUP BY txn_hour ORDER BY txn_hour
    """).fetchall()

    # By day
    by_day = conn.execute("""
        SELECT day_of_week,
               SUM(is_fraud) as fraud,
               COUNT(*) as total
        FROM transactions GROUP BY day_of_week ORDER BY day_of_week
    """).fetchall()

    # Amount distribution (10 buckets)
    amounts = conn.execute("SELECT amount FROM transactions").fetchall()
    conn.close()

    return {
        "total": total,
        "fraud": fraud_cnt,
        "legit": legit_cnt,
        "fraud_rate": fraud_rate,
        "avg_amount": round(avg_amt, 2),
        "avg_risk": round(avg_risk, 2),
        "by_merchant": [dict(r) for r in by_merchant],
        "by_type":     [dict(r) for r in by_type],
        "by_hour":     [dict(r) for r in by_hour],
        "by_day":      [dict(r) for r in by_day],
        "amounts":     [r[0] for r in amounts],
    }
