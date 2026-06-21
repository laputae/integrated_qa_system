# -*-coding:utf-8-*-
"""
SQLite-based file ingestion tracker.

Tracks which files have been ingested, their content hashes, and processing status.
Enables incremental document processing by categorizing files as:
  NEW / MODIFIED / UNCHANGED / DELETED
"""
import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional

from base import logger

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".ppt", ".pptx", ".jpg", ".png", ".md"}


class IngestionTracker:
    """SQLite-backed tracker for file ingestion state."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT UNIQUE NOT NULL,
                file_name   TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                file_size   INTEGER DEFAULT 0,
                file_mtime  REAL DEFAULT 0.0,
                chunk_count INTEGER DEFAULT 0,
                status      TEXT DEFAULT 'active',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingested_status "
            "ON ingested_files(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingested_doc_id "
            "ON ingested_files(doc_id)"
        )
        self.conn.commit()

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """SHA-256 hex digest of file bytes, streamed in 64KB chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_doc_id(file_path: str) -> str:
        """Derive a stable, deterministic doc_id from the normalized file path."""
        abs_path = os.path.abspath(file_path)
        normalized = os.path.normpath(abs_path).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def scan_directory(self, directory_path: str) -> Dict[str, List[Dict]]:
        """Scan directory and categorize each supported file.

        Returns:
            dict with keys: "new", "modified", "unchanged", "deleted"
        """
        result: Dict[str, List[Dict]] = {
            "new": [],
            "modified": [],
            "unchanged": [],
            "deleted": [],
        }

        seen_paths: set = set()

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                norm_path = os.path.normpath(file_path).lower()
                seen_paths.add(norm_path)

                try:
                    content_hash = self.compute_file_hash(file_path)
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    file_mtime = stat.st_mtime
                    doc_id = self.compute_doc_id(file_path)
                except (FileNotFoundError, IOError, PermissionError) as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
                    continue

                prev = self._lookup(file_path)

                entry = {
                    "file_path": file_path,
                    "file_name": file,
                    "file_size": file_size,
                    "file_mtime": file_mtime,
                    "content_hash": content_hash,
                    "doc_id": doc_id,
                }

                if prev is None:
                    result["new"].append(entry)
                elif prev["content_hash"] != content_hash:
                    result["modified"].append(entry)
                else:
                    result["unchanged"].append(entry)

        # Find DELETED: active records not on disk
        for record in self._get_all_active():
            norm_path = os.path.normpath(record["file_path"]).lower()
            if norm_path not in seen_paths:
                result["deleted"].append({
                    "file_path": record["file_path"],
                    "doc_id": record["doc_id"],
                    "file_name": record["file_name"],
                })

        logger.info(
            f"扫描结果: {len(result['new'])} 新增, "
            f"{len(result['modified'])} 修改, "
            f"{len(result['unchanged'])} 未变, "
            f"{len(result['deleted'])} 已删除"
        )
        return result

    def mark_ingested(
        self,
        file_path: str,
        content_hash: str,
        doc_id: str,
        file_size: int = 0,
        file_mtime: float = 0.0,
        chunk_count: int = 0,
    ) -> None:
        """Mark a file as successfully ingested (upsert)."""
        now = datetime.now().isoformat()
        file_path = os.path.normpath(file_path).lower()
        file_name = os.path.basename(file_path)
        self.conn.execute(
            """
            INSERT INTO ingested_files
                (file_path, file_name, content_hash, doc_id, file_size,
                 file_mtime, chunk_count, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash = excluded.content_hash,
                file_size    = excluded.file_size,
                file_mtime   = excluded.file_mtime,
                chunk_count  = excluded.chunk_count,
                status       = 'active',
                updated_at   = excluded.updated_at
            """,
            (file_path, file_name, content_hash, doc_id,
             file_size, file_mtime, chunk_count, now, now),
        )
        self.conn.commit()

    def mark_deleted(self, doc_id: str) -> None:
        """Mark a document as deleted (soft-delete, sets status='deleted')."""
        now = datetime.now().isoformat()
        self.conn.execute(
            "UPDATE ingested_files SET status='deleted', chunk_count=0, "
            "updated_at=? WHERE doc_id=?",
            (now, doc_id),
        )
        self.conn.commit()

    def get_doc_id(self, file_path: str) -> Optional[str]:
        """Get the doc_id for a known file path, or None."""
        row = self._lookup(file_path)
        return row["doc_id"] if row else None

    def get_chunk_count(self, doc_id: str) -> int:
        """Get the number of chunks previously stored for a doc_id."""
        cursor = self.conn.execute(
            "SELECT chunk_count FROM ingested_files "
            "WHERE doc_id=? AND status='active'",
            (doc_id,),
        )
        row = cursor.fetchone()
        return row["chunk_count"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # ---- internal helpers ----

    def _lookup(self, file_path: str) -> Optional[Dict]:
        """Look up a file by path, returning a dict or None."""
        norm_path = os.path.normpath(file_path).lower()
        cursor = self.conn.execute(
            "SELECT content_hash, doc_id, file_size, file_mtime, status "
            "FROM ingested_files WHERE file_path=?",
            (norm_path,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _get_all_active(self) -> List[Dict]:
        """Get all active (non-deleted) records."""
        cursor = self.conn.execute(
            "SELECT file_path, doc_id, file_name "
            "FROM ingested_files WHERE status='active'"
        )
        return [dict(row) for row in cursor.fetchall()]
