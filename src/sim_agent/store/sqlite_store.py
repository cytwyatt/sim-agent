from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sim_agent.types import PaperRecord


class SQLiteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    top_n INTEGER NOT NULL,
                    years INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    run_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    year INTEGER,
                    simulation_type TEXT NOT NULL,
                    summary TEXT,
                    json_blob TEXT NOT NULL,
                    PRIMARY KEY (run_id, paper_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def store_run(
        self,
        run_id: str,
        topic: str,
        created_at: str,
        top_n: int,
        years: int,
        records: list[PaperRecord],
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, topic, created_at, top_n, years)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, topic, created_at, top_n, years),
            )
            for record in records:
                blob = json.dumps(record.to_dict(), ensure_ascii=False)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO papers
                    (run_id, paper_id, title, year, simulation_type, summary, json_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        record.paper_metadata.paper_id,
                        record.paper_metadata.title,
                        record.paper_metadata.year,
                        record.simulation_type,
                        record.summary,
                        blob,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
