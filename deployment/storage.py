import sqlite3
from typing import List

class Storage:
    def __init__(self, db_path: str= 'predictions.db'):
        """
        Initializes SQLite connection and ensures the log table exists.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        """
        Creates the logs table if it doesn't already exist.
        """
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            text TEXT,
            label TEXT,
            confidence REAL,
            keywords TEXT
            );
            ''')
        self.conn.commit()

    def log(self, timestamp:str, text:str, label: str, confidence:float, keywords: List[str]):
        """
        Inserts a new prediction record into the logs table.
        """
        self.conn.execute(
            '''
            INSERT INTO logs(timestamp, text, label, confidence, keywords)
            VALUES (?,?,?,?,?);
            '''
            ,(timestamp, text, label, confidence, ','.join(keywords)))
        self.conn.commit()

# Sanity check
if __name__ == '__main__':
    s = Storage()
    s.log("2025-05-26T20:30:00", "Test complaint narrative", "Neutral", 0.92, ["test", "complaint"])
    tables = s.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    print("SQLite tables:", tables)
        